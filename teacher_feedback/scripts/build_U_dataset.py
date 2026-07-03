#!/usr/bin/env python3
"""
构建U组预训练数据集: 在Q组训练数据基础上注入实体追踪对。

策略:
  - 从原始139,513篇文档中随机移除 N_REMOVE 篇
  - 生成 N_ENTITY 个实体追踪场景 (仅chosen=正确答案，用于预训练)
  - 将实体对tokenize后以文档形式追加
  - 总词数仍控制在100M以内

论据:
  - S组实验: info-mask令Entity-tracking从40.42降至35.01 (-5.4分)
  - U组目标: 用Q config + entity pairs弥补entity tracking缺陷
  - 实体对chosen tokens约224K，仅需移除234篇文档(<0.17%)

输出: /data0/lexi/babyllava/data_new/train_U_entity.bin
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

# ── 路径 ─────────────────────────────────────────────────────────────────
SRC_BIN  = Path("/data0/lexi/babyllava/data_new/train_100M_optimal.bin")
TOK_PATH = Path("/data0/lexi/babyllava/babylm_eng_baseline_tokenizer/tokenizer.json")
OUT_BIN  = Path("/data0/lexi/babyllava/data_new/train_U_entity.bin")
ENTITY_OUT = Path("/data0/lexi/babyllava/teacher_feedback/data/u_entity_chosen.jsonl")

# ── 实体追踪场景生成 ───────────────────────────────────────────────────
# 与eval场景不同的容器/物品（避免数据泄漏）
CONTAINERS = ["bag", "drawer", "shelf", "basket", "trunk", "pocket", "locker",
               "cabinet", "crate", "bin", "sack", "jar", "bowl", "plate", "tray"]
ITEMS      = ["apple", "key", "coin", "pen", "book", "ring", "ball", "card",
               "stone", "leaf", "note", "clip", "chip", "button", "thread",
               "fork", "lens", "cube", "disc", "pin"]


def generate_entity_scene(n_containers: int = 3, include_rejected: bool = False):
    """生成一个实体追踪场景，只返回正确答案(chosen)用于预训练"""
    containers = random.sample(CONTAINERS, n_containers)
    items_in = {c: [] for c in containers}
    used_items = random.sample(ITEMS, n_containers + 1)

    for i, c in enumerate(containers):
        items_in[c].append(used_items[i])

    n_moves = random.randint(1, 2)
    moves = []
    for _ in range(n_moves):
        src = random.choice(containers)
        if not items_in[src]:
            continue
        item = random.choice(items_in[src])
        dst_choices = [c for c in containers if c != src]
        if not dst_choices:
            continue
        dst = random.choice(dst_choices)
        items_in[src].remove(item)
        items_in[dst].append(item)
        moves.append((item, src, dst))

    if not moves:
        return None

    state = {c: [used_items[i]] for i, c in enumerate(containers)}
    for item, src, dst in moves:
        state[src].remove(item)
        state[dst].append(item)

    parts = []
    for i, c in enumerate(containers):
        parts.append(f"The {c} contains the {used_items[i]}.")
    for item, src, dst in moves:
        parts.append(f"We move the {item} from the {src} to the {dst}.")

    query_container = random.choice(containers)
    final_items = state[query_container]
    parts.append(f"What does the {query_container} contain?")
    prefix = " ".join(parts)

    if final_items:
        chosen = f"{prefix} The {query_container} contains the {final_items[0]}."
    else:
        chosen = f"{prefix} The {query_container} is empty."

    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_entity",  type=int, default=5000)
    parser.add_argument("--n_remove",  type=int, default=None,
                        help="移除文档数量，默认自动计算（与实体对等词数）")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",    default=str(OUT_BIN))
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 加载原始数据 ────────────────────────────────────────────────────
    print(f"加载 {SRC_BIN} ...", flush=True)
    docs = torch.load(str(SRC_BIN), map_location="cpu")
    print(f"原始文档数: {len(docs)}", flush=True)

    total_orig_tokens = sum(len(d) for d in docs)
    print(f"原始总token数: {total_orig_tokens:,}", flush=True)

    # ── 生成实体对 ────────────────────────────────────────────────────
    print(f"\n生成 {args.n_entity} 个实体追踪场景 ...", flush=True)
    tok = Tokenizer.from_file(str(TOK_PATH))
    entity_docs = []
    entity_texts = []
    attempts = 0

    while len(entity_texts) < args.n_entity and attempts < args.n_entity * 5:
        attempts += 1
        n_c = random.choice([2, 3, 4])
        text = generate_entity_scene(n_c)
        if text is None:
            continue
        entity_texts.append(text)

    print(f"生成场景: {len(entity_texts)}")

    # tokenize
    for text in entity_texts:
        ids = tok.encode(text).ids
        if len(ids) < 5:
            continue
        entity_docs.append(torch.tensor(ids, dtype=torch.int16))

    entity_token_count = sum(len(d) for d in entity_docs)
    print(f"实体对总token数: {entity_token_count:,}", flush=True)

    # 保存实体文本便于检查
    ENTITY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ENTITY_OUT, "w") as f:
        for t in entity_texts:
            f.write(json.dumps({"text": t}) + "\n")
    print(f"实体文本保存至 {ENTITY_OUT}")

    # ── 计算需要移除的文档数 ────────────────────────────────────────────
    if args.n_remove is None:
        # 移除与实体对相同数量的token所对应的文档
        avg_doc_len = total_orig_tokens / len(docs)
        n_remove = max(1, round(entity_token_count / avg_doc_len))
    else:
        n_remove = args.n_remove

    print(f"\n将随机移除 {n_remove} 篇文档 (共{len(docs)}篇，占{n_remove/len(docs):.3%})")

    # 随机选择要移除的文档索引
    remove_idx = set(random.sample(range(len(docs)), n_remove))
    kept_docs = [d for i, d in enumerate(docs) if i not in remove_idx]

    kept_token_count = sum(len(d) for d in kept_docs)
    print(f"保留文档: {len(kept_docs)}", flush=True)
    print(f"保留token数: {kept_token_count:,}", flush=True)

    # ── 合并 ──────────────────────────────────────────────────────────
    all_docs = kept_docs + entity_docs
    random.shuffle(all_docs)

    final_token_count = sum(len(d) for d in all_docs)
    print(f"\n最终文档数: {len(all_docs)}", flush=True)
    print(f"最终token数: {final_token_count:,}", flush=True)
    print(f"相比原始: {(final_token_count - total_orig_tokens):+,} tokens", flush=True)

    # ── 保存 ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_docs, str(out_path))
    print(f"\nU组数据集已保存: {out_path}")
    print(f"  文档数: {len(all_docs):,}")
    print(f"  Token数: {final_token_count:,}")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  文件大小: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
