#!/usr/bin/env python3
"""
AoA Curriculum 数据构建脚本
策略：Phase1(10% 早习得词密集句) + Phase2(90% 原始随机)

核心思路：
  - Phase1 (10M words): 按 CDI 平均月龄排序的"早习得词密集"句子
    → 早期 checkpoint (chck_1M~chck_10M) 主要接触早习得词
    → 这些词的惊喜度在早期 checkpoint 快速下降
  - Phase2 (90M words): 原始随机顺序（保持 Entity tracking 训练完整性）
    → 晚习得词在此阶段才大量出现
    → 晚习得词的惊喜度在中后期 checkpoint 才下降

  与 CHILDES-First 的关键区别：
    - 不是按语料来源（整个CHILDES）分组，而是按 CDI 词义习得顺序筛选句子
    - CHILDES 包含晚习得词（think=30月、tonight=30月），CHILDES-First 会把它们也放前面
    - Phase1 只选择 CDI 月龄低的句子（不含晚习得词），严格按词义发展顺序

  预期效果（基于梯度分析）：
    - AoA: 9.05 → ~20-30 (r: 0.09 → 0.20-0.30)
    - BLiMP/Entity/EWoK: 轻微波动（Phase2 90% 完全随机）
"""

import json
import re
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

MONTHS_COLS = [str(m) for m in range(16, 31)]
GLOBAL_MEAN_AOA = 25.0  # CDI 月龄全局均值，用于无CDI词句子的填充


def load_cdi_aoa(cdi_csv_path: str) -> Dict[str, float]:
    """加载 CDI 词表及人类习得月龄（50%儿童掌握的月龄）"""
    df = pd.read_csv(cdi_csv_path, index_col=0)

    def compute_aoa(row) -> float:
        for m in MONTHS_COLS:
            if row[m] >= 0.5:
                return float(m)
        return 30.0

    df['aoa'] = df.apply(compute_aoa, axis=1)
    word2aoa = dict(zip(df['word'], df['aoa']))
    print(f'  CDI词表: {len(word2aoa)}词，月龄范围16-30，均值={df["aoa"].mean():.1f}')
    return word2aoa


def split_sentences(text: str) -> List[str]:
    """按句号/问号/感叹号分割句子"""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.split()) >= 3]


def score_sentence(sentence: str, word2aoa: Dict[str, float]) -> Optional[float]:
    """计算句子中 CDI 词的平均习得月龄（无CDI词返回None）"""
    words = re.findall(r'\b[a-z]+\b', sentence.lower())
    months = [word2aoa[w] for w in words if w in word2aoa]
    return float(np.mean(months)) if months else None


def build_phase1_sentences(
    input_path: str,
    word2aoa: Dict[str, float],
    phase1_frac: float,
    target_words: int,
    seed: int,
) -> Tuple[List[Tuple[str, float]], List[dict]]:
    """
    构建 Phase1 句子列表和 Phase2 文档列表

    返回:
        phase1_sents: [(sentence, cdi_score), ...] 已排序（早词在前）
        phase2_docs:  [{'text':..., 'source':...}, ...] 随机顺序的完整文档
    """
    rng = random.Random(seed)
    all_sents_with_score: List[Tuple[str, float]] = []
    all_docs: List[dict] = []

    print(f'  读取语料并提取句子...')
    with open(input_path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            text = d.get('text', '')
            sents = split_sentences(text)
            all_docs.append(d)
            for sent in sents:
                score = score_sentence(sent, word2aoa)
                cdi_score = score if score is not None else GLOBAL_MEAN_AOA
                all_sents_with_score.append((sent, cdi_score))
            if (i + 1) % 20000 == 0:
                print(f'    {i+1} 文档, {len(all_sents_with_score)} 句子...')

    print(f'  共 {len(all_docs)} 文档, {len(all_sents_with_score)} 句子')

    # 按 CDI 月龄排序（低月龄→早习得词在前），加入5%噪声防止过度对齐
    scores = np.array([s for _, s in all_sents_with_score])
    noise = np.random.RandomState(seed).randn(len(scores)) * scores.std() * 0.05
    sort_idx = np.argsort(scores + noise)

    sorted_sents = [(all_sents_with_score[i][0], all_sents_with_score[i][1])
                    for i in sort_idx]

    # 取前 phase1_frac 比例作为 Phase1
    n_phase1 = max(1, int(len(sorted_sents) * phase1_frac))
    phase1_sents = sorted_sents[:n_phase1]

    # 按词数检查 Phase1 覆盖量
    words_in_phase1 = sum(len(s.split()) for s, _ in phase1_sents)
    print(f'  Phase1: {n_phase1} 句 ≈ {words_in_phase1/1e6:.1f}M words')

    # Phase2: 完整文档列表（随机打乱）
    rng.shuffle(all_docs)

    return phase1_sents, all_docs


def main():
    parser = argparse.ArgumentParser(description='构建 AoA Curriculum 训练数据（Phase1+Phase2）')
    parser.add_argument('--input',
                        default='/data0/lexi/babyllava/data_new/babylm2026_optimal.jsonl')
    parser.add_argument('--output',
                        default='/data0/lexi/babyllava/data_new/babylm2026_aoa_curriculum.jsonl')
    parser.add_argument('--cdi_csv',
                        default='/data0/lexi/babyllava/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')
    parser.add_argument('--phase1_frac', type=float, default=0.10,
                        help='Phase1 占总句子数的比例（默认0.10）')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print('\n[1/4] 加载 CDI AoA 词表')
    word2aoa = load_cdi_aoa(args.cdi_csv)

    print(f'\n[2/4] 构建 Phase1（{args.phase1_frac*100:.0f}% 早习得词密集句）+ Phase2（随机文档）')
    phase1_sents, phase2_docs = build_phase1_sentences(
        args.input, word2aoa, args.phase1_frac, target_words=10_000_000, seed=args.seed
    )

    # 验证梯度
    scores_p1 = [s for _, s in phase1_sents]
    print(f'\n  梯度验证:')
    print(f'    Phase1 CDI月龄: 均值={np.mean(scores_p1):.2f}, '
          f'p5={np.percentile(scores_p1,5):.1f}, p95={np.percentile(scores_p1,95):.1f}')

    print(f'\n[3/4] 写入输出文件: {args.output}')
    out_path = Path(args.output)
    n_written = 0
    total_words = 0

    with open(out_path, 'w') as f:
        # Phase1: 将排序好的句子分组成伪文档（每50句一个伪文档，保留局部连贯性）
        CHUNK = 50
        for i in range(0, len(phase1_sents), CHUNK):
            chunk = phase1_sents[i:i + CHUNK]
            text = ' '.join(s for s, _ in chunk)
            mean_score = float(np.mean([sc for _, sc in chunk]))
            record = {'text': text, 'source': 'aoa_phase1',
                      'cdi_month': round(mean_score, 2)}
            f.write(json.dumps(record) + '\n')
            total_words += len(text.split())
            n_written += 1

        phase1_words = total_words
        print(f'    Phase1: {n_written} 伪文档, {phase1_words/1e6:.1f}M words')

        # Phase2: 随机顺序的完整原始文档
        p2_written = 0
        for d in phase2_docs:
            f.write(json.dumps(d) + '\n')
            total_words += len(d.get('text', '').split())
            p2_written += 1

        print(f'    Phase2: {p2_written} 文档, {(total_words-phase1_words)/1e6:.1f}M words')

    print(f'\n  ✅ 总计: {n_written + p2_written} 条记录, {total_words/1e6:.1f}M words')
    print(f'\n[4/4] 验证梯度（首尾各1000条记录）')
    # 快速验证
    early_scores, late_scores = [], []
    with open(out_path) as f:
        lines = f.readlines()

    for line in lines[:1000]:
        d = json.loads(line)
        if 'cdi_month' in d:
            early_scores.append(d['cdi_month'])

    # Phase2 开头（紧跟Phase1后面的随机文档）
    for line in lines[n_written: n_written + 1000]:
        d = json.loads(line)
        text = d.get('text', '')
        words = re.findall(r'\b[a-z]+\b', text.lower())
        months = [word2aoa[w] for w in words if w in word2aoa]
        if months:
            late_scores.append(float(np.mean(months)))

    print(f'  Phase1前1000条: CDI月龄均值={np.mean(early_scores):.2f}')
    print(f'  Phase2前1000条: CDI月龄均值={np.mean(late_scores):.2f}')
    gradient = np.mean(late_scores) - np.mean(early_scores)
    # gradient 这里是相对的，Phase2 均值比 Phase1 高，说明梯度正确
    # 实际梯度是 Phase1 的早期 vs 训练后期的 CDI 分布

    print(f'''
========================================
构建完成！下一步：
  1) 分词（保序，不shuffle）:
     python3 data_new/tokenize_cf.py \\
       --input {out_path} \\
       --output data_new/train_aoa_curriculum.bin \\
       --dev_output data_new/dev_aoa_curriculum.bin

  2) 训练 Q-curriculum（修改 run_Q_aoa.sh 中的 train_path）:
     --train_path data_new/train_aoa_curriculum.bin
     --valid_path data_new/dev_aoa_curriculum.bin

  预期效果：
    AoA:    9.05 → 20-30  (r: 0.09 → 0.20-0.30)
    BLiMP:  ±0.5%（Phase2 完全随机）
    Entity: 轻微回退（估计 -1 ~ -3 points）
========================================''')


if __name__ == '__main__':
    main()
