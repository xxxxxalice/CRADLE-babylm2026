#!/usr/bin/env python3
"""
教师标注: 用 Qwen3-8B-Instruct 过滤偏好对。

规则合规:
  Qwen3-8B-Instruct 属于Qwen 3家族，参数量8B < 9B上限，符合BabyLM 2026规则。
  教师只输出二元标签 (A/B)，不暴露概率/隐藏状态/权重。

最优实践 (基于2024-2025年LLM-as-judge研究):
  1. 双向swap消除位置偏差: 每对做两次推理 (A-B 和 B-A)
     - 两次一致 → 保留 (teacher_certain=True)
     - 两次不一致 → 标为uncertain，可选丢弃
  2. CoT先于答案: 先解释理由，再给 A/B
  3. 分类型prompt: 语法类 vs 实体追踪类

输出字段:
  teacher_correct:  bool, 教师两次都认为chosen更好
  teacher_certain:  bool, 两次swap结果是否一致
  swap1_answer:     'A'/'B'  第一次 (chosen=A, rejected=B)
  swap2_answer:     'A'/'B'  第二次 (chosen=B, rejected=A)
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 8B纯文本模型路径（下载中）
QWEN_PATH_8B = "/data0/lexi/models/Qwen3-8B"
# 4B VL模型路径（已存在，可立即使用）
QWEN_PATH_4B = "/data0/models/Qwen3-VL-4B-Instruct"

QWEN_PATH = QWEN_PATH_8B  # 默认，可通过--model_path覆盖
DATA_DIR    = Path(__file__).parent.parent / "data"
INPUT_FILE  = DATA_DIR / "all_pairs_raw.jsonl"
OUTPUT_FILE = DATA_DIR / "teacher_labeled.jsonl"


def load_model(model_path: str, device: str = "cuda"):
    print(f"加载教师模型 {model_path} ...", flush=True)
    import json, os
    cfg_path = os.path.join(model_path, "config.json")
    model_type = json.load(open(cfg_path)).get("model_type", "")

    if "vl" in model_type.lower() or "VL" in model_path:
        # Qwen3-VL 系列：用 Qwen3VLForConditionalGeneration + Qwen3VLProcessor
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        proc = Qwen3VLProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model.eval()
        print(f"教师模型加载完成 (Qwen3-VL, text-only模式)", flush=True)
        return model, proc, True
    else:
        # 纯文本模型 (Qwen3-8B等)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
        print(f"教师模型加载完成 (纯文本)", flush=True)
        return model, tokenizer, False


def build_prompt(text_a: str, text_b: str, pair_type: str) -> str:
    if "entity" in pair_type:
        task_desc = (
            "Read the two passages below. Each describes objects being moved between containers "
            "and ends with a statement about the final contents. "
            "Which passage correctly tracks where the object ends up?\n"
            "Briefly explain your reasoning in one sentence, then answer with only 'A' or 'B'."
        )
    else:
        task_desc = (
            "Which of the two sentences below is more grammatically natural English?\n"
            "Briefly explain your reasoning in one sentence, then answer with only 'A' or 'B'."
        )
    return (
        f"{task_desc}\n\n"
        f"A: {text_a}\n\n"
        f"B: {text_b}\n\n"
        f"Reasoning and answer:"
    )


@torch.inference_mode()
def query_teacher(model, proc, is_vl: bool, prompt: str, device: str) -> str:
    """推理一次，返回 'A'/'B'/'?' """
    messages = [{"role": "user", "content": prompt}]
    # enable_thinking=False: 禁用 Qwen3 思考模式，避免 <think> 块中的 A/B 干扰答案解析
    try:
        text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if is_vl:
        inputs = proc(text=[text], return_tensors="pt").to(device)
        eos_id = proc.tokenizer.eos_token_id
    else:
        inputs = proc(text, return_tensors="pt").to(device)
        eos_id = proc.eos_token_id
    out = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=eos_id,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    tok = proc.tokenizer if is_vl else proc
    response = tok.decode(new_tokens, skip_special_tokens=True).strip()
    matches = re.findall(r'\b([AB])\b', response.upper())
    return matches[-1] if matches else "?"


def judge_pair(model, proc, is_vl: bool, chosen: str, rejected: str,
               pair_type: str, device: str) -> dict:
    """
    双向swap判断。
    Pass1: A=chosen,   B=rejected → 期望回答 A
    Pass2: A=rejected, B=chosen   → 期望回答 B
    """
    ans1 = query_teacher(model, proc, is_vl,
                         build_prompt(chosen, rejected, pair_type), device)
    ans2 = query_teacher(model, proc, is_vl,
                         build_prompt(rejected, chosen, pair_type), device)

    teacher_correct_1 = (ans1 == "A")
    teacher_correct_2 = (ans2 == "B")

    consistent      = (teacher_correct_1 == teacher_correct_2)
    teacher_correct = teacher_correct_1 and teacher_correct_2
    teacher_certain = consistent and (ans1 != "?" and ans2 != "?")

    return {
        "swap1_answer":    ans1,
        "swap2_answer":    ans2,
        "teacher_correct": teacher_correct,
        "teacher_certain": teacher_certain,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default=str(INPUT_FILE))
    parser.add_argument("--output",       default=str(OUTPUT_FILE))
    parser.add_argument("--model_path",   default=None,
                        help="模型路径，默认自动选择(8B优先，fallback到4B VL)")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--max",          type=int, default=None)
    parser.add_argument("--certain_only", action="store_true",
                        help="只输出teacher_certain=True的对")
    args = parser.parse_args()

    # 自动选择模型
    import os
    if args.model_path:
        model_path = args.model_path
    elif os.path.isdir(QWEN_PATH_8B) and any(
        f.endswith(".safetensors") for f in os.listdir(QWEN_PATH_8B)
    ):
        model_path = QWEN_PATH_8B
        print("自动选择: Qwen3-8B (已下载完成)")
    else:
        model_path = QWEN_PATH_4B
        print("自动选择: Qwen3-VL-4B-Instruct (8B尚未就绪，使用4B)")

    pairs = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line.strip()))
    if args.max:
        pairs = pairs[:args.max]
    print(f"待标注: {len(pairs)} 对", flush=True)

    model, proc, is_vl = load_model(model_path, args.device)

    n_correct = 0
    n_wrong   = 0
    n_split   = 0
    n_ambig   = 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(out_path, "w") as fout:
        for i, pair in enumerate(pairs):
            chosen   = pair["chosen"]
            rejected = pair["rejected"]
            ptype    = pair.get("type", "grammar")

            result = judge_pair(model, proc, is_vl, chosen, rejected, ptype, args.device)

            if result["swap1_answer"] == "?" or result["swap2_answer"] == "?":
                n_ambig += 1
                continue

            if result["teacher_correct"]:
                n_correct += 1
            elif not result["teacher_certain"]:
                n_split += 1
            else:
                n_wrong += 1

            if args.certain_only and not result["teacher_certain"]:
                continue

            fout.write(json.dumps({
                "chosen":          chosen,
                "rejected":        rejected,
                "type":            ptype,
                **result,
            }) + "\n")

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                total   = n_correct + n_wrong + n_split
                eta     = (elapsed / (i + 1)) * (len(pairs) - i - 1)
                print(
                    f"[{i+1}/{len(pairs)}] "
                    f"agree={n_correct}({n_correct/max(1,total):.0%})  "
                    f"disagree={n_wrong}({n_wrong/max(1,total):.0%})  "
                    f"split={n_split}  ambig={n_ambig}  "
                    f"ETA={eta/60:.1f}min",
                    flush=True,
                )

    total = n_correct + n_wrong + n_split
    print(f"\n=== 标注完成 ===")
    print(f"  两次都认为chosen更好: {n_correct:5d} ({n_correct/max(1,total):.1%})")
    print(f"  两次都认为rejected好: {n_wrong:5d} ({n_wrong/max(1,total):.1%})")
    print(f"  两次不一致(split):    {n_split:5d} ({n_split/max(1,total):.1%})")
    print(f"  模糊/跳过:            {n_ambig:5d}")
    print(f"输出: {out_path}")

    type_stats = {}
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            t = r.get("type", "?")
            if t not in type_stats:
                type_stats[t] = {"agree": 0, "disagree": 0, "split": 0}
            if r["teacher_correct"]:
                type_stats[t]["agree"] += 1
            elif r["teacher_certain"]:
                type_stats[t]["disagree"] += 1
            else:
                type_stats[t]["split"] += 1

    print("\n各类型统计 (教师同意chosen / 总):")
    for t, s in sorted(type_stats.items()):
        total_t = s["agree"] + s["disagree"] + s["split"]
        print(f"  {t:22s}: agree={s['agree']:4d} "
              f"disagree={s['disagree']:4d} split={s['split']:4d} "
              f"({s['agree']/max(1,total_t):.1%})")


if __name__ == "__main__":
    main()
