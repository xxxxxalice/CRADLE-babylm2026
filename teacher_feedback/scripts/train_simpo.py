#!/usr/bin/env python3
"""
偏好优化训练 - 支持 SimPER / SimPO / KTO 三种方法。

基于2024-2025年论文的最优实践:

【方法选择】
  SimPER (默认, 推荐):
    - Meng et al. arXiv:2502.00883, ICLR 2025
    - 无超参数，比SimPO高4.9分(AlpacaEval2)
    - loss = -exp(avg_lp_chosen) + exp(avg_lp_rejected)
    - 等价于: 最小化chosen困惑度 + 最大化rejected困惑度

  SimPO (备选):
    - Meng et al. arXiv:2405.14734, NeurIPS 2024
    - 关键: β=2.0~3.0 (not 0.1!), γ/β≈0.5
    - loss = -logsigmoid(β*(avg_lp_chosen - avg_lp_rejected - γ))

  KTO (非成对数据场景):
    - Ethayarajh et al. arXiv:2402.01306
    - 只需单条(text, label)，无需成对
    - 适合教师标注预算有限时

【MNTP打分】(GPT-BERT专用)
  GPTBertForMaskedLM.forward返回全位置logits (B,L,V)
  avg_logprob = mean log P(token_i | 双向全上下文), i in non-pad
  比单向CLM更准确，与eval pipeline的--backend mntp一致

【防崩溃保护】(基于SLIME/AlphaPO 2025)
  --anchor_weight: 对chosen添加NLL锚定损失，防止chosen logprob下滑
  chosen_nll = -avg_lp_chosen.mean()  (维持原始语言模型能力)
  total = pref_loss + anchor_weight * chosen_nll

用法:
  torchrun --standalone --nproc_per_node=4 train_simpo.py \
    --model_path .../hf_s_infomask_75mlm_stepXXXX \
    --data_path  .../teacher_labeled.jsonl \
    --output_dir .../checkpoints/simper_v1 \
    --loss_type simper --lr 5e-5 --batch_size 64 --max_steps 1000
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tokenizers import Tokenizer
from transformers import AutoConfig

# GPT-BERT eval模型注册
# 将 gpt-bert/ 加入 sys.path，作为包根目录（pretraining/ 有 __init__.py）
_gptbert_root = str(Path(__file__).parent.parent.parent / "gpt-bert")
if _gptbert_root not in sys.path:
    sys.path.insert(0, _gptbert_root)
try:
    from pretraining.modeling_gpt_bert_eval import GPTBertForMaskedLM, GPTBertConfig
    from transformers import AutoModelForMaskedLM
    AutoConfig.register("gpt_bert", GPTBertConfig)
    AutoModelForMaskedLM.register(GPTBertConfig, GPTBertForMaskedLM)
except Exception as e:
    raise ImportError(f"无法加载GPT-BERT模型: {e}")

TOK_PATH = "/data0/lexi/babyllava/babylm_eng_baseline_tokenizer/tokenizer.json"

MODEL_CODE_FILES = [
    "model_extra.py", "modeling_gpt_bert_eval.py",
    "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json",
]


def save_hf_checkpoint(model, src_model_path: Path, dst_path: Path):
    """保存HF格式checkpoint，复制eval pipeline所需的代码文件"""
    dst_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst_path))
    for fname in MODEL_CODE_FILES:
        src = src_model_path / fname
        if src.exists():
            shutil.copy2(str(src), str(dst_path / fname))


# ════════════════════════════════════════════════════════════════════════
# 数据集 (支持成对和非成对两种格式)
# ════════════════════════════════════════════════════════════════════════

class PrefDataset(Dataset):
    """
    支持两种格式:
      成对: {"chosen": "...", "rejected": "...", "teacher_correct": true}
      非成对(KTO): {"text": "...", "label": true/false}  (label=True表示好样本)
    """
    def __init__(self, jsonl_path: str, loss_type: str = "simper",
                 teacher_only: bool = True, min_margin_ratio: float = 0.0):
        self.loss_type   = loss_type
        self.paired      = []   # (chosen_text, rejected_text)
        self.unpaired    = []   # (text, is_good)

        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line.strip())

                if "text" in r:
                    # KTO格式
                    self.unpaired.append((r["text"], bool(r.get("label", True))))
                else:
                    # 成对格式
                    if teacher_only and not r.get("teacher_correct", True):
                        continue
                    self.paired.append((r["chosen"], r["rejected"]))

        if loss_type == "kto" and self.unpaired:
            n = len(self.unpaired)
            n_good = sum(1 for _, g in self.unpaired if g)
            print(f"KTO数据集: {n} 条 (好样本={n_good}, 坏样本={n - n_good})")
        else:
            print(f"偏好数据集: {len(self.paired)} 对 "
                  f"(loss={loss_type}, teacher_only={teacher_only})")

    def __len__(self):
        if self.loss_type == "kto" and self.unpaired:
            return len(self.unpaired)
        return len(self.paired)

    def __getitem__(self, idx):
        if self.loss_type == "kto" and self.unpaired:
            return self.unpaired[idx]  # (text, is_good)
        return self.paired[idx]        # (chosen, rejected)


def collate_paired(batch, tokenizer, max_length: int, pad_id: int):
    """成对格式collate: 返回 (c_ids, c_mask, r_ids, r_mask)"""
    chosen_texts, rejected_texts = zip(*batch)

    def encode_batch(texts):
        encs = [tokenizer.encode(t).ids[:max_length] for t in texts]
        max_len = max(len(e) for e in encs)
        ids  = torch.zeros(len(encs), max_len, dtype=torch.long)
        mask = torch.zeros(len(encs), max_len, dtype=torch.long)
        for i, e in enumerate(encs):
            ids[i, :len(e)]  = torch.tensor(e, dtype=torch.long)
            mask[i, :len(e)] = 1
        return ids, mask

    c_ids, c_mask = encode_batch(chosen_texts)
    r_ids, r_mask = encode_batch(rejected_texts)
    return c_ids, c_mask, r_ids, r_mask


def collate_unpaired(batch, tokenizer, max_length: int, pad_id: int):
    """KTO格式collate: 返回 (ids, mask, labels)"""
    texts, labels = zip(*batch)
    encs = [tokenizer.encode(t).ids[:max_length] for t in texts]
    max_len = max(len(e) for e in encs)
    ids  = torch.zeros(len(encs), max_len, dtype=torch.long)
    mask = torch.zeros(len(encs), max_len, dtype=torch.long)
    for i, e in enumerate(encs):
        ids[i, :len(e)]  = torch.tensor(e, dtype=torch.long)
        mask[i, :len(e)] = 1
    lbls = torch.tensor(labels, dtype=torch.float)
    return ids, mask, lbls


# ════════════════════════════════════════════════════════════════════════
# MNTP 打分 (GPT-BERT 双向上下文)
# ════════════════════════════════════════════════════════════════════════

def mntp_avg_logprob(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    GPTBertForMaskedLM 返回全位置logits (B, L, V)。
    avg_lp[b] = mean over non-pad positions of log P(token_i | context)
    与eval pipeline的 --backend mntp 打分方式一致。
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (B,L,V)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # (B,L)
    mask_f = attention_mask.float()
    return (token_logprobs * mask_f).sum(-1) / mask_f.sum(-1).clamp(min=1)     # (B,)


# ════════════════════════════════════════════════════════════════════════
# 损失函数
# ════════════════════════════════════════════════════════════════════════

def loss_simper(chosen_lp: torch.Tensor, rejected_lp: torch.Tensor) -> tuple:
    """
    SimPER (arXiv:2502.00883, ICLR 2025) - 无超参数版本。
    L = -exp(avg_lp_chosen) + exp(avg_lp_rejected)
      = -(1/ppl_chosen) + (1/ppl_rejected)
    即: 最小化chosen的困惑度，最大化rejected的困惑度。
    """
    loss = (-chosen_lp.exp() + rejected_lp.exp()).mean()
    with torch.no_grad():
        acc = (chosen_lp > rejected_lp).float().mean()
        raw_margin = (chosen_lp - rejected_lp).mean()
    return loss, {
        "loss":       loss.item(),
        "accuracy":   acc.item(),
        "raw_margin": raw_margin.item(),
        "chosen_lp":  chosen_lp.mean().item(),
        "rejected_lp": rejected_lp.mean().item(),
    }


def loss_simpo(chosen_lp: torch.Tensor, rejected_lp: torch.Tensor,
               beta: float = 2.5, gamma: float = 1.0) -> tuple:
    """
    SimPO (arXiv:2405.14734, NeurIPS 2024).
    最优超参 (来自消融): β=2.0~3.0, γ/β≈0.5 (即γ=1.0~1.5 when β=2.5).
    L = -logsigmoid(β*(avg_lp_chosen - avg_lp_rejected - γ))
    """
    margin = beta * (chosen_lp - rejected_lp - gamma)
    loss   = -F.logsigmoid(margin).mean()
    with torch.no_grad():
        acc          = (chosen_lp > rejected_lp + gamma).float().mean()
        mean_margin  = margin.mean()
    return loss, {
        "loss":        loss.item(),
        "accuracy":    acc.item(),
        "mean_margin": mean_margin.item(),
        "chosen_lp":   chosen_lp.mean().item(),
        "rejected_lp": rejected_lp.mean().item(),
    }


def loss_kto(model, ids: torch.Tensor, mask: torch.Tensor,
             labels: torch.Tensor, beta: float = 0.1) -> tuple:
    """
    KTO (arXiv:2402.01306) 简化版 - 非成对数据。
    好样本: -logsigmoid(β * avg_lp)  (鼓励高log prob)
    坏样本: -logsigmoid(-β * avg_lp)  (惩罚高log prob)
    """
    lp = mntp_avg_logprob(model, ids, mask)
    good  = labels.bool()
    bad   = ~good
    loss_good = -F.logsigmoid( beta * lp[good]).mean()  if good.any() else 0.0
    loss_bad  = -F.logsigmoid(-beta * lp[bad]).mean()   if bad.any()  else 0.0
    loss = (loss_good + loss_bad) / 2
    with torch.no_grad():
        acc_good = (lp[good] > 0).float().mean().item() if good.any() else float('nan')
    return loss, {
        "loss":     loss.item(),
        "acc_good": acc_good,
        "lp_good":  lp[good].mean().item() if good.any() else float('nan'),
        "lp_bad":   lp[bad].mean().item()  if bad.any()  else float('nan'),
    }


# ════════════════════════════════════════════════════════════════════════
# 主训练循环
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    required=True)
    parser.add_argument("--data_path",     required=True)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--loss_type",     default="simper",
                        choices=["simper", "simpo", "kto"],
                        help="simper=无超参推荐; simpo=需β,γ; kto=非成对数据")
    # SimPO超参 (simper时忽略)
    parser.add_argument("--beta",          type=float, default=2.5,
                        help="SimPO β，文献最优2.0~3.0 (不是0.1!)")
    parser.add_argument("--gamma_margin",  type=float, default=1.0,
                        help="SimPO γ，文献最优γ/β≈0.5，即β=2.5时γ=1.0")
    # 防崩溃锚定 (基于SLIME/AlphaPO)
    parser.add_argument("--anchor_weight", type=float, default=0.1,
                        help="NLL锚定损失权重，防止chosen logprob崩溃")
    # 训练配置
    parser.add_argument("--lr",            type=float, default=5e-5,
                        help="比预训练lr小100倍(7e-3→5e-5)，防止灾难遗忘")
    parser.add_argument("--batch_size",    type=int,   default=64,   help="全局batch size")
    parser.add_argument("--max_steps",     type=int,   default=1000)
    parser.add_argument("--max_length",    type=int,   default=128)
    parser.add_argument("--save_every",    type=int,   default=250)
    parser.add_argument("--log_every",     type=int,   default=25)
    parser.add_argument("--warmup_steps",  type=int,   default=50)
    parser.add_argument("--teacher_only",  action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    # ── 分布式初始化 ──────────────────────────────────────────────────
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    torch.manual_seed(args.seed + rank)

    # ── Tokenizer ────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(TOK_PATH)
    pad_id = tokenizer.token_to_id("<pad>")

    # ── 模型 ─────────────────────────────────────────────────────────
    log(f"加载模型 {args.model_path}")
    model = GPTBertForMaskedLM.from_pretrained(args.model_path, trust_remote_code=True)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    log("模型加载完成")

    # ── 数据集 ────────────────────────────────────────────────────────
    dataset = PrefDataset(args.data_path, args.loss_type, args.teacher_only)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    local_batch = max(1, args.batch_size // world_size)

    if args.loss_type == "kto" and dataset.unpaired:
        cfn = lambda b: collate_unpaired(b, tokenizer, args.max_length, pad_id)
    else:
        cfn = lambda b: collate_paired(b, tokenizer, args.max_length, pad_id)

    loader = DataLoader(
        dataset, batch_size=local_batch, sampler=sampler,
        num_workers=2, collate_fn=cfn, pin_memory=True, drop_last=True,
    )

    # ── 优化器 ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_fn(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    method_desc = {
        "simper": "SimPER (arXiv:2502.00883) — 无超参数",
        "simpo":  f"SimPO (arXiv:2405.14734) — β={args.beta}, γ={args.gamma_margin}",
        "kto":    f"KTO (arXiv:2402.01306) — β={args.beta}",
    }[args.loss_type]

    log(f"\n偏好优化训练开始")
    log(f"  方法:         {method_desc}")
    log(f"  锚定权重:     anchor_weight={args.anchor_weight}")
    log(f"  lr={args.lr}, global_batch={args.batch_size}, steps={args.max_steps}")

    # ── 训练循环 ──────────────────────────────────────────────────────
    step    = 0
    epoch   = 0
    running = {}
    n_accum = 0

    while step < args.max_steps:
        epoch += 1
        sampler.set_epoch(epoch)

        for batch in loader:
            if step >= args.max_steps:
                break

            model.train()
            optimizer.zero_grad()

            if args.loss_type == "kto":
                ids, mask, labels = [x.to(device) for x in batch]
                loss, metrics = loss_kto(model, ids, mask, labels, args.beta)
            else:
                c_ids, c_mask, r_ids, r_mask = [x.to(device) for x in batch]
                chosen_lp   = mntp_avg_logprob(model, c_ids, c_mask)
                rejected_lp = mntp_avg_logprob(model, r_ids, r_mask)

                if args.loss_type == "simper":
                    loss, metrics = loss_simper(chosen_lp, rejected_lp)
                else:
                    loss, metrics = loss_simpo(chosen_lp, rejected_lp,
                                               args.beta, args.gamma_margin)

                # 锚定损失: 防止chosen logprob崩溃 (基于SLIME/AlphaPO发现)
                if args.anchor_weight > 0:
                    anchor = -chosen_lp.mean()
                    loss   = loss + args.anchor_weight * anchor
                    metrics["anchor_nll"] = anchor.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step    += 1
            n_accum += 1
            for k, v in metrics.items():
                running[k] = running.get(k, 0) + (v if isinstance(v, float) else float(v))

            if step % args.log_every == 0 and rank == 0:
                avg  = {k: v / n_accum for k, v in running.items()}
                lr_now = optimizer.param_groups[0]["lr"]
                parts = [f"step {step:4d}/{args.max_steps}",
                         f"loss={avg.get('loss', 0):.4f}",
                         f"acc={avg.get('accuracy', avg.get('acc_good', 0)):.3f}"]
                if "chosen_lp" in avg:
                    parts.append(f"c_lp={avg['chosen_lp']:.3f}")
                    parts.append(f"r_lp={avg['rejected_lp']:.3f}")
                if "raw_margin" in avg:
                    parts.append(f"margin={avg['raw_margin']:.4f}")
                parts.append(f"lr={lr_now:.1e}")
                print("  ".join(parts), flush=True)
                running = {}
                n_accum = 0

            if step % args.save_every == 0 and rank == 0:
                save_path = Path(args.output_dir) / f"{args.loss_type}_step{step}"
                save_hf_checkpoint(model.module, Path(args.model_path), save_path)
                log(f"[step {step}] 保存 → {save_path}")

    if rank == 0:
        final_path = Path(args.output_dir) / f"{args.loss_type}_step{step}_final"
        save_hf_checkpoint(model.module, Path(args.model_path), final_path)
        log(f"\n训练完成！最终模型: {final_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
