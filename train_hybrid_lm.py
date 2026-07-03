import os
import shutil
import json
import argparse
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM, get_cosine_schedule_with_warmup
from transformers.trainer_pt_utils import LengthGroupedSampler
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
import wandb
from functools import partial

# =======================================================
# 🛡️ 架构师核级环境注入 (显存碎片整理与核心加速)
# ======================================================= 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =======================================================
# 🧩 核心架构：纯正混合单双向底座 (Hybrid Causal-Masked LM)
# =======================================================
class PureHybridBabyLM(nn.Module):
    def __init__(self, text_config):
        super().__init__()
        text_config._attn_implementation = "sdpa" 
        self.model = LlamaForCausalLM(text_config) 
        self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, base_attention_mask, is_causal_mode=True):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        extended_attention_mask = base_attention_mask[:, None, None, :].expand(batch_size, 1, seq_len, seq_len).to(dtype=torch.float32)
        
        if is_causal_mode:
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
            extended_attention_mask = extended_attention_mask.masked_fill(~causal_mask, 0.0)
        
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask = extended_attention_mask.to(torch.bfloat16)

        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=extended_attention_mask, 
            output_hidden_states=False 
        )
        return outputs.logits

# =======================================================
# 📊 纯文本极速数据加载器 (关闭死板 Padding)
# =======================================================
class PureTextDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        print(f"⚡ 正在将语料加载至内存: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f: 
            self.data = json.load(f)
        print(f"✅ 语料加载完成！共 {len(self.data)} 条数据。")
        
        # 预计算所有样本的近似长度，用于后续的分组采样
        print("📏 正在预计算序列长度，用于分布式智能分组...")
        self.lengths = [len(f"{item.get('q', '')} {item.get('a', '')}".split() if item.get("is_qa") else item.get("text", "").split()) for item in self.data]
        
    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text_content = f"{item.get('q', '')} {item.get('a', '')}" if item.get("is_qa") else item.get("text", "")
            
        encoded = self.tokenizer(
            text_content + self.tokenizer.eos_token, 
            truncation=True, 
            max_length=self.max_seq_len, 
            padding=False,  # 🚀 核心：关闭固定填充
            return_tensors="pt"
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

# =======================================================
# ⚡ 智能动态填充调度器 (Collate Function)
# =======================================================
def dynamic_collate_fn(batch, pad_token_id):
    """动态对齐当前 Batch 内的最长序列，配合分组采样，完美压榨算力"""
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    return padded_input_ids, padded_attention_mask

# =======================================================
# 🛡️ 容灾与安全落盘
# =======================================================
def save_checkpoint(accelerator, model, optimizer, scheduler, total_tokens, path):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        try:
            backup_path = path.replace(".pt", "_BACKUP.pt")
            if os.path.exists(path): shutil.move(path, backup_path) 
            
            unwrapped_model = accelerator.unwrap_model(model)
            model_state_dict = unwrapped_model._orig_mod.state_dict() if hasattr(unwrapped_model, "_orig_mod") else unwrapped_model.state_dict()
                
            state = {
                "model": model_state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "total_tokens": total_tokens
            }
            torch.save(state, path)
            print(f"\n💾 进度存档成功: {total_tokens/1e6:.2f}M | 文件: {os.path.basename(path)}")
        except Exception as e: print(f"⚠️ 保存异常: {e}")

# =======================================================
# 🚀 极限挑战主循环
# =======================================================
def run_training(args):
    os.makedirs(args.out_model_dir, exist_ok=True)
    
    process_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=4))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulation_steps, 
        mixed_precision="bf16", 
        kwargs_handlers=[process_kwargs]
    )

    if accelerator.is_main_process:
        wandb.init(project="BabyLM-V4-PureText", name=f"Text-1B-Ratio_{args.causal_to_masked_ratio}", config=vars(args))

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    if getattr(tokenizer, "mask_token_id", None) is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id

    config = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=768, num_hidden_layers=10, 
        num_attention_heads=12, max_position_embeddings=args.max_seq_len
    )
    model = PureHybridBabyLM(config).to(accelerator.device)
    model.model.resize_token_embeddings(len(tokenizer))

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    dataset = PureTextDataset(args.data_json, tokenizer, args.max_seq_len)
    
    # 🏆 核心：挂载分布式长度分组采样器
    sampler = LengthGroupedSampler(
        batch_size=args.batch_size,
        dataset=dataset,
        lengths=dataset.lengths,
        model_input_name="input_ids"
    )
    
    collate_fn = partial(dynamic_collate_fn, pad_token_id=pad_token_id)
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, 
        num_workers=4, prefetch_factor=2, pin_memory=True, drop_last=True,
        collate_fn=collate_fn
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=0.1, betas=(0.9, 0.95), fused=True)
    
    estimated_total_steps = int((args.max_training_words) / (args.batch_size * args.accumulation_steps * accelerator.num_processes * 512))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(estimated_total_steps * args.warmup_ratio), num_training_steps=estimated_total_steps)

    total_tokens_seen = 0
    latest_path = os.path.join(args.out_model_dir, "V4_ROLLING_LATEST.pt")
    
    if os.path.exists(latest_path):
        loaded_ckpt = torch.load(latest_path, map_location="cpu")
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, "_orig_mod"): unwrapped_model._orig_mod.load_state_dict(loaded_ckpt["model"])
        else: unwrapped_model.load_state_dict(loaded_ckpt["model"])
            
        optimizer.load_state_dict(loaded_ckpt["optimizer"])
        scheduler.load_state_dict(loaded_ckpt["scheduler"])
        total_tokens_seen = loaded_ckpt.get("total_tokens", 0)
        del loaded_ckpt; gc.collect()
        if accelerator.is_main_process: print(f"🔍 已从 {total_tokens_seen/1e6:.2f}M 无缝恢复。")

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    next_rolling_save = (int(total_tokens_seen) // args.rolling_save_interval + 1) * args.rolling_save_interval

    try:
        model.train()
        exported_milestones = set()
        
        for epoch in range(args.epochs):
            pbar = tqdm(dataloader, desc=f"🏆 V4 Sprint (Epoch {epoch+1})", disable=not accelerator.is_main_process)
            
            for input_ids, attn_mask in pbar:
                
                local_valid_tokens = attn_mask.sum()
                global_valid_tokens = accelerator.gather(local_valid_tokens.unsqueeze(0)).sum().item()
                total_tokens_seen += global_valid_tokens
                
                if total_tokens_seen >= args.max_training_words: break

                with accelerator.accumulate(model):
                    mode_flag = torch.tensor([1 if torch.rand(1).item() < args.causal_to_masked_ratio else 0], device=accelerator.device)
                    if torch.distributed.is_initialized(): torch.distributed.broadcast(mode_flag, src=0)
                    is_causal_batch = bool(mode_flag.item() == 1)

                    if is_causal_batch:
                        model_inputs = input_ids
                        labels = input_ids.clone()
                    else:
                        model_inputs = input_ids.clone()
                        prob_matrix = torch.full(input_ids.shape, args.mask_prob, device=accelerator.device)
                        masked_indices = torch.bernoulli(prob_matrix).bool() & (attn_mask == 1)
                        masked_indices[:, 0] = False 
                        model_inputs[masked_indices] = mask_token_id
                        
                        labels = torch.full_like(input_ids, -100)
                        labels[masked_indices] = input_ids[masked_indices] 

                    logits = model(model_inputs, attn_mask, is_causal_mode=is_causal_batch)
                        
                    if is_causal_batch:
                        valid_logits = logits[:, :-1, :].contiguous().view(-1, config.vocab_size)
                        valid_labels = labels[:, 1:].contiguous().view(-1)
                        valid_labels[valid_labels == pad_token_id] = -100
                    else:
                        valid_logits = logits.view(-1, config.vocab_size)
                        valid_labels = labels.view(-1)
                    
                    valid_mask = (valid_labels != -100)
                    
                    if valid_mask.any():
                        final_logits = valid_logits[valid_mask]  
                        final_labels = valid_labels[valid_mask]        
                        gen_loss = F.cross_entropy(final_logits, final_labels)
                        
                        log_z = torch.logsumexp(final_logits, dim=-1) 
                        z_loss = (log_z ** 2).mean()
                        total_loss = gen_loss + args.z_loss_weight * z_loss
                    else:
                        total_loss = valid_logits.sum() * 0.0

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        optimizer.zero_grad()
                        continue

                    accelerator.backward(total_loss)
                    if accelerator.sync_gradients: accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                if accelerator.sync_gradients and accelerator.is_main_process:
                    mode_str = "CAUS" if is_causal_batch else "MASK"
                    pbar.set_postfix({"W(M)": f"{total_tokens_seen/1e6:.1f}", "Mode": mode_str, "Loss": f"{gen_loss.item():.3f}"})
                    wandb.log({
                        "total_loss": total_loss.item(), 
                        "gen_loss": gen_loss.item(), 
                        "lr": scheduler.get_last_lr()[0], 
                        "words_million": total_tokens_seen/1e6,
                        "is_causal": int(is_causal_batch)
                    })

                if total_tokens_seen >= next_rolling_save:
                    save_checkpoint(accelerator, model, optimizer, scheduler, total_tokens_seen, latest_path)
                    next_rolling_save += args.rolling_save_interval

                for milestone in args.hf_export_milestones:
                    if total_tokens_seen >= milestone and milestone not in exported_milestones:
                        save_checkpoint(accelerator, model, optimizer, scheduler, total_tokens_seen, latest_path)
                        if accelerator.is_main_process:
                            save_dir = os.path.join(args.out_model_dir, f"babyvlm_hf_{milestone/1e6:.0f}M")
                            unwrapped = accelerator.unwrap_model(model)
                            if hasattr(unwrapped, "_orig_mod"): unwrapped = unwrapped._orig_mod 
                            unwrapped.model.save_pretrained(save_dir)
                            tokenizer.save_pretrained(save_dir)
                        exported_milestones.add(milestone)

            save_checkpoint(accelerator, model, optimizer, scheduler, total_tokens_seen, latest_path)
            torch.cuda.empty_cache() 
            
            if total_tokens_seen >= args.max_training_words:
                if accelerator.is_main_process: print(f"\n🎉 目标达成！10亿次有效曝光完成！")
                break

    except KeyboardInterrupt:
        if accelerator.is_main_process: print("\n🚨 迫降机制启动，安全落盘...")
        save_checkpoint(accelerator, model, optimizer, scheduler, total_tokens_seen, os.path.join(args.out_model_dir, "V4_EMERGENCY_SAVE.pt"))
    finally:
        if accelerator.is_main_process: wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BabyLM 纯文本打榜版底座架构")
    
    parser.add_argument("--data_json", type=str, default="/data0/lexi/babyllava/babylm_100M_PURE_TEXT_FINAL.json")
    parser.add_argument("--tokenizer_path", type=str, default="/data0/lexi/babyllava/babylm_eng_baseline_tokenizer")
    parser.add_argument("--out_model_dir", type=str, required=True)
    
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150)
    
    parser.add_argument("--max_training_words", type=int, default=1_000_000_000)
    parser.add_argument("--hf_export_milestones", type=int, nargs='+', default=[500_000_000, 750_000_000, 1_000_000_000])
    parser.add_argument("--rolling_save_interval", type=int, default=5_000_000)
    
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    
    parser.add_argument("--causal_to_masked_ratio", type=float, default=0.5)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--z_loss_weight", type=float, default=0.0001)

    args = parser.parse_args()
    run_training(args)