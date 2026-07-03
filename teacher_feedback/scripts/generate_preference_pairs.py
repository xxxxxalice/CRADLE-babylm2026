#!/usr/bin/env python3
"""
生成偏好对 (chosen, rejected) 用于SimPO训练。

两类对:
1. 语法对: 原句(chosen) vs 规则扰动句(rejected)
   - 使用训练数据中的真实句子，避免使用eval数据
2. 实体追踪对: 生成新的box场景 (模板与eval不同)
   - chosen: 正确追踪实体位置
   - rejected: 错误答案(混淆位置)

输出: data/grammar_pairs.jsonl + data/entity_pairs.jsonl
      每行: {"chosen": "...", "rejected": "...", "type": "grammar|entity"}
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

# ── 路径配置 ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_BIN = Path("/data0/lexi/babyllava/data_new/train_100M_optimal.bin")
TOK_PATH  = Path("/data0/lexi/babyllava/babylm_eng_baseline_tokenizer/tokenizer.json")


# ════════════════════════════════════════════════════════════════════════
# 1. 语法扰动规则
# ════════════════════════════════════════════════════════════════════════

# 3单动词: 常见词 → 去掉 -s / -es
SVA_RULES = [
    (r'\b(is)\b', 'are'),
    (r'\b(was)\b', 'were'),
    (r'\b(has)\b', 'have'),
    (r'\b(does)\b', 'do'),
    (r'\b(goes)\b', 'go'),
    (r'\b(runs)\b', 'run'),
    (r'\b(comes)\b', 'come'),
    (r'\b(makes)\b', 'make'),
    (r'\b(takes)\b', 'take'),
    (r'\b(gives)\b', 'give'),
    (r'\b(gets)\b', 'get'),
    (r'\b(says)\b', 'say'),
    (r'\b(knows)\b', 'know'),
    (r'\b(thinks)\b', 'think'),
    (r'\b(sees)\b', 'see'),
    (r'\b(wants)\b', 'want'),
    (r'\b(needs)\b', 'need'),
    (r'\b(likes)\b', 'like'),
    (r'\b(seems)\b', 'seem'),
    (r'\b(becomes)\b', 'become'),
    (r'\b(shows)\b', 'show'),
    (r'\b(means)\b', 'mean'),
    (r'\b(remains)\b', 'remain'),
    (r'\b(appears)\b', 'appear'),
    (r'\b(contains)\b', 'contain'),
    (r'\b(includes)\b', 'include'),
    (r'\b(requires)\b', 'require'),
    (r'\b(provides)\b', 'provide'),
    (r'\b(begins)\b', 'begin'),
    (r'\b(ends)\b', 'end'),
]

# 复数名词 → 去掉 -s: 只处理明确数词修饰的情况
PLURAL_RULES = [
    (r'\b(two|three|four|five|six|seven|eight|nine|ten|several|many|few|multiple)\s+(\w+s)\b',
     lambda m: m.group(1) + ' ' + m.group(2)[:-1]),  # two cats → two cat
]

# 冠词删除
ARTICLE_DEL_RULES = [
    (r'\b(a) ([aeiouAEIOU]\w+)\b', r'\2'),      # a apple → apple
    (r'\b(the) (\w+)\b', r'\2'),                 # the cat → cat (随机触发)
]

# 时态错误: 过去式 → 现在时
TENSE_RULES = [
    (r'\b(went)\b', 'go'),
    (r'\b(came)\b', 'come'),
    (r'\b(said)\b', 'say'),
    (r'\b(took)\b', 'take'),
    (r'\b(made)\b', 'make'),
    (r'\b(gave)\b', 'give'),
    (r'\b(got)\b', 'get'),
    (r'\b(knew)\b', 'know'),
    (r'\b(saw)\b', 'see'),
    (r'\b(thought)\b', 'think'),
    (r'\b(found)\b', 'find'),
    (r'\b(told)\b', 'tell'),
    (r'\b(became)\b', 'become'),
    (r'\b(showed)\b', 'show'),
    (r'\b(began)\b', 'begin'),
    (r'\b(kept)\b', 'keep'),
    (r'\b(held)\b', 'hold'),
    (r'\b(left)\b', 'leave'),
    (r'\b(felt)\b', 'feel'),
    (r'\b(brought)\b', 'bring'),
]


def apply_sva_corruption(sentence: str) -> str | None:
    """应用主谓一致错误，优先选在主语后面的动词"""
    random.shuffle(SVA_RULES)
    for pattern, replacement in SVA_RULES:
        m = re.search(pattern, sentence)
        if m:
            return sentence[:m.start()] + replacement + sentence[m.end():]
    return None


def apply_tense_corruption(sentence: str) -> str | None:
    random.shuffle(TENSE_RULES)
    for pattern, replacement in TENSE_RULES:
        m = re.search(pattern, sentence)
        if m:
            return sentence[:m.start()] + replacement + sentence[m.end():]
    return None


def apply_article_corruption(sentence: str) -> str | None:
    """删除 'a' 或 (低概率) 'the'"""
    # 只删除 'a' 开头不定冠词，避免语义过于模糊
    pattern = r'\b(a) ([aeiouAEIOU]\w+)\b'
    m = re.search(pattern, sentence)
    if m:
        return sentence[:m.start()] + m.group(2) + sentence[m.end():]
    # 低概率删除 the
    if random.random() < 0.3:
        m2 = re.search(r'\b(the) (\w+)\b', sentence)
        if m2:
            return sentence[:m2.start()] + m2.group(2) + sentence[m2.end():]
    return None


CORRUPTION_FUNCS = [
    ('sva',     apply_sva_corruption),
    ('tense',   apply_tense_corruption),
    ('article', apply_article_corruption),
]


def corrupt_sentence(sentence: str) -> tuple[str, str] | None:
    """返回 (corruption_type, corrupted_sentence) 或 None"""
    fns = CORRUPTION_FUNCS[:]
    random.shuffle(fns)
    for name, fn in fns:
        result = fn(sentence)
        if result and result != sentence:
            return name, result
    return None


# ════════════════════════════════════════════════════════════════════════
# 2. 实体追踪场景生成
#    使用不同于eval的容器/物品名，避免数据泄漏
# ════════════════════════════════════════════════════════════════════════

CONTAINERS = ["bag", "drawer", "shelf", "basket", "trunk", "pocket", "locker",
               "cabinet", "crate", "bin", "sack", "jar", "bowl", "plate", "tray"]
ITEMS      = ["apple", "key", "coin", "pen", "book", "ring", "ball", "card",
               "stone", "leaf", "note", "clip", "chip", "button", "thread",
               "fork", "lens", "cube", "disc", "pin"]
NAMES      = ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry"]


def generate_entity_pair(n_containers: int = 3) -> dict:
    """
    生成一个实体追踪偏好对。
    chosen: 正确答案 (追踪到最终位置)
    rejected: 错误答案 (混淆位置)
    """
    containers = random.sample(CONTAINERS, n_containers)
    # 分配物品到容器
    items_in = {c: [] for c in containers}
    used_items = random.sample(ITEMS, n_containers + 1)

    for i, c in enumerate(containers):
        items_in[c].append(used_items[i])

    # 随机做 1-2 次移动
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

    # 从初始状态重放移动，确定最终状态
    state = {c: [used_items[i]] for i, c in enumerate(containers)}
    for item, src, dst in moves:
        state[src].remove(item)
        state[dst].append(item)

    # 构建前缀文本
    prefix_parts2 = []
    for i, c in enumerate(containers):
        prefix_parts2.append(f"The {c} contains the {used_items[i]}.")
    for item, src, dst in moves:
        prefix_parts2.append(f"We move the {item} from the {src} to the {dst}.")

    # 选一个容器提问
    query_container = random.choice(containers)
    final_items = state[query_container]
    prefix_parts2.append(f"What does the {query_container} contain?")
    prefix = " ".join(prefix_parts2)

    if final_items:
        chosen_answer = f"The {query_container} contains the {final_items[0]}."
        # rejected: 说它是空的
        rejected_answer = f"The {query_container} is empty."
    else:
        chosen_answer = f"The {query_container} is empty."
        # rejected: 随机说一个错误的物品
        wrong_item = random.choice([x for x in used_items if x not in (state.get(query_container) or [])])
        rejected_answer = f"The {query_container} contains the {wrong_item}."

    return {
        "chosen":   prefix + " " + chosen_answer,
        "rejected": prefix + " " + rejected_answer,
        "type":     "entity",
    }


# ════════════════════════════════════════════════════════════════════════
# 3. 主程序
# ════════════════════════════════════════════════════════════════════════

def load_training_sentences(n_sentences: int, min_len: int = 15, max_len: int = 100) -> list[str]:
    """从训练数据中随机抽取句子"""
    print(f"加载训练数据 {TRAIN_BIN} ...", flush=True)
    tok = Tokenizer.from_file(str(TOK_PATH))
    docs = torch.load(str(TRAIN_BIN))
    print(f"共 {len(docs)} 篇文档", flush=True)

    sentences = []
    idx = list(range(len(docs)))
    random.shuffle(idx)

    for i in idx:
        doc = docs[i]
        text = tok.decode([int(x) for x in doc.tolist() if int(x) > 0])
        # 按句号切分
        parts = re.split(r'(?<=[.!?])\s+', text)
        for p in parts:
            p = p.strip()
            words = p.split()
            if min_len <= len(words) <= max_len:
                sentences.append(p)
        if len(sentences) >= n_sentences * 5:  # 多采一些，留过滤空间
            break

    random.shuffle(sentences)
    return sentences[:n_sentences * 5]


def token_length_ratio(s1: str, s2: str) -> float:
    """两句长度之比，越接近1越好（扰动不应大幅改变长度）"""
    n1, n2 = len(s1.split()), len(s2.split())
    return min(n1, n2) / max(n1, n2) if max(n1, n2) > 0 else 1.0


def edit_distance_ratio(s1: str, s2: str) -> float:
    """简单词级别编辑距离比例 (基于词袋差异)"""
    w1, w2 = set(s1.lower().split()), set(s2.lower().split())
    diff = len(w1.symmetric_difference(w2))
    total = len(w1.union(w2))
    return diff / total if total > 0 else 0.0


def generate_grammar_pairs(sentences: list[str], target: int) -> list[dict]:
    """
    生成语法偏好对，加入margin质量过滤。

    质量过滤标准 (基于"Less is More" arXiv:2502.14560):
    - 编辑距离比例 0.03~0.30: 扰动既可见(>3%)又不过于破坏性(<30%)
    - 长度比例 > 0.8: 扰动不应大幅改变序列长度
    - 排除近似重复对 (edit_ratio < 0.03)
    """
    pairs = []
    corruption_counts = {}
    n_filtered = 0

    for sent in sentences:
        if len(pairs) >= target:
            break
        result = corrupt_sentence(sent)
        if result is None:
            continue
        ctype, corrupted = result
        if len(corrupted.split()) < 6 or corrupted == sent:
            continue

        # Margin质量过滤
        edit_ratio   = edit_distance_ratio(sent, corrupted)
        length_ratio = token_length_ratio(sent, corrupted)

        if edit_ratio < 0.03:   # 扰动太小，几乎看不出
            n_filtered += 1
            continue
        if edit_ratio > 0.35:   # 扰动太大，句子面目全非
            n_filtered += 1
            continue
        if length_ratio < 0.80:  # 长度变化过大
            n_filtered += 1
            continue

        pairs.append({
            "chosen":   sent,
            "rejected": corrupted,
            "type":     f"grammar_{ctype}",
        })
        corruption_counts[ctype] = corruption_counts.get(ctype, 0) + 1

    print(f"语法对生成统计: {corruption_counts}  过滤丢弃: {n_filtered}", flush=True)
    return pairs


def main():
    parser = argparse.ArgumentParser()
    # 文献 "Less is More" (arXiv:2502.14560): 高质量10%数据 > 全量
    # BabyLM场景: 500-2000高质量对已足够偏好优化收敛
    parser.add_argument("--n_grammar",  type=int, default=15000, help="语法对数量(生成后教师过滤)")
    parser.add_argument("--n_entity",   type=int, default=5000,  help="实体追踪对数量")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── 生成语法对 ──────────────────────────────────────────────────────
    print(f"\n=== 生成语法对 (目标 {args.n_grammar}) ===")
    sentences = load_training_sentences(args.n_grammar)
    grammar_pairs = generate_grammar_pairs(sentences, args.n_grammar)

    grammar_out = DATA_DIR / "grammar_pairs.jsonl"
    with open(grammar_out, "w") as f:
        for p in grammar_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"语法对保存到 {grammar_out} ({len(grammar_pairs)} 条)")

    # ── 生成实体追踪对 ──────────────────────────────────────────────────
    print(f"\n=== 生成实体追踪对 (目标 {args.n_entity}) ===")
    entity_pairs = []
    attempts = 0
    while len(entity_pairs) < args.n_entity and attempts < args.n_entity * 3:
        attempts += 1
        n_c = random.choice([2, 3, 4])
        p = generate_entity_pair(n_c)
        if p:
            entity_pairs.append(p)

    entity_out = DATA_DIR / "entity_pairs.jsonl"
    with open(entity_out, "w") as f:
        for p in entity_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"实体对保存到 {entity_out} ({len(entity_pairs)} 条)")

    # ── 合并 ──────────────────────────────────────────────────────────
    all_pairs = grammar_pairs + entity_pairs
    random.shuffle(all_pairs)
    all_out = DATA_DIR / "all_pairs_raw.jsonl"
    with open(all_out, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"\n总计 {len(all_pairs)} 对，合并保存到 {all_out}")
    print("下一步: 运行 teacher_label.py 进行教师筛选")


if __name__ == "__main__":
    main()
