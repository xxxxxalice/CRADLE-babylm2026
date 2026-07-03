"""
从 babylm2026_optimal.jsonl 中精确削减 Wikipedia 词数，
使预训练语料从 100.00M 减到 99.32M，
为 SimPER 的 0.68M 新文本腾出空间，实现总曝光恰好 = 100.00M。

用法:
    python3 trim_wikipedia.py

输出:
    babylm2026_compliant.jsonl  (99.32M words, 有 source 字段)
    data_new/train_compliant.bin (tokenized, 可直接用于训练)
"""

import json
from pathlib import Path

INPUT  = Path("/data0/lexi/babyllava/data_new/babylm2026_optimal.jsonl")
OUTPUT = Path("/data0/lexi/babyllava/data_new/babylm2026_compliant.jsonl")

SIMPER_BUDGET   = 680_000      # SimPER 实际新文本：~0.68M words
TARGET_PRETRAIN = 100_000_000 - SIMPER_BUDGET  # 99,320,000 words
WIKI_CUT        = SIMPER_BUDGET  # 从 Wikipedia 削减同等量

def count_words(text):
    return len(text.split())

# ── 第一遍：收集所有 Wikipedia 文档及其词数 ──────────────────
print("第一遍：统计各来源词数...")
wiki_docs   = []  # (行号, 词数, 原始json字符串)
other_total = 0
wiki_total  = 0

with open(INPUT) as f:
    for i, line in enumerate(f):
        obj   = json.loads(line)
        src   = obj.get("source", "")
        words = count_words(obj["text"])
        if src == "wikipedia":
            wiki_docs.append((i, words, line.rstrip()))
            wiki_total += words
        else:
            other_total += words
        if i % 200000 == 0:
            print(f"  {i:,} 行...", end="\r", flush=True)

print(f"\n  非Wikipedia: {other_total/1e6:.3f}M words")
print(f"  Wikipedia:   {wiki_total/1e6:.3f}M words")
print(f"  合计:        {(other_total+wiki_total)/1e6:.3f}M words")

# ── 从 Wikipedia 末尾删除直到削减够 0.68M ────────────────────
# Wikipedia 是"填充源"——按流式下载顺序排列
# 从末尾删除对语义覆盖影响最小（开头的文章通常更常见）
removed_words = 0
keep_count    = len(wiki_docs)

for i in range(len(wiki_docs) - 1, -1, -1):
    if removed_words >= WIKI_CUT:
        break
    _, w, _ = wiki_docs[i]
    removed_words += w
    keep_count = i  # 保留到此索引（不含）

kept_wiki_words = sum(w for (_, w, _) in wiki_docs[:keep_count])
final_total     = other_total + kept_wiki_words

print(f"\n── 削减计划 ──")
print(f"  Wikipedia 保留: {keep_count:,} 篇 / {len(wiki_docs):,} 篇")
print(f"  Wikipedia 词数: {wiki_total/1e6:.3f}M → {kept_wiki_words/1e6:.3f}M")
print(f"  削减量:         {removed_words/1e6:.3f}M words")
print(f"  新预训练总量:   {final_total/1e6:.3f}M words")
print(f"  加上SimPER:     {(final_total+SIMPER_BUDGET)/1e6:.3f}M words")
print(f"  合规状态:       {'✓ ≤ 100M' if final_total+SIMPER_BUDGET <= 100_000_000 else '✗ 超出'}")

confirm = input("\n确认写入新语料文件? [y/N] ").strip().lower()
if confirm != "y":
    print("已取消。")
    exit()

# ── 第二遍：写出合规语料 ─────────────────────────────────────
print(f"\n写入 {OUTPUT} ...")
keep_wiki_lines = {line_no for (line_no, _, _) in wiki_docs[:keep_count]}
written_total   = 0
wiki_line_nos   = {line_no for (line_no, _, _) in wiki_docs}

with open(INPUT) as fin, open(OUTPUT, "w") as fout:
    for i, line in enumerate(fin):
        if i in wiki_line_nos:
            if i in keep_wiki_lines:
                fout.write(line)
                written_total += count_words(json.loads(line)["text"])
        else:
            fout.write(line)
            written_total += count_words(json.loads(line)["text"])
        if i % 200000 == 0:
            print(f"  {i:,} 行, {written_total/1e6:.3f}M words...", end="\r", flush=True)

print(f"\n完成！写入 {written_total/1e6:.3f}M words 到 {OUTPUT}")
print(f"\n下一步：重新 tokenize 并训练")
print(f"  python3 data_new/tokenize_to_bin.py \\")
print(f"    --input {OUTPUT} \\")
print(f"    --train_out data_new/train_compliant.bin \\")
print(f"    --dev_out   data_new/dev_compliant.bin")
print(f"\n  然后修改 train_hybrid_lm.py 的 --data_json 指向 babylm2026_compliant.jsonl")
