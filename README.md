# CRADLE: Curriculum and Reasoning via Age-of-acquisition-Driven Lexical Enhancement for BabyLM 2026

**BabyLM 2026 Strict Track** · TextAvg **47.80** (surpasses leaderboard #1 by +0.30)

> Paper: *CRADLE: Curriculum and Reasoning via Age-of-acquisition-Driven Lexical Enhancement for BabyLM 2026*  
> Author: Gan Wang, Xi'an Jiaotong-Liverpool University  
> Model: [AliceAndNoob/babylm2026-strict-gptbert-simper](https://huggingface.co/AliceAndNoob/babylm2026-strict-gptbert-simper)

---

## Overview

CRADLE combines two complementary innovations for data-efficient language learning under the 100M-word constraint:

1. **CDI Lexical Curriculum** — Sentences with earliest-acquired vocabulary (bottom 10th percentile of MacArthur-Bates CDI AoA norms) are placed in Phase 1 (first 10% of training steps), giving the model child-like early exposure. Raises AoA from **9.05 → 17.24** (+8.19).

2. **SimPER v2 Post-training** — 29,996 targeted preference pairs including novel `move_contents` (7-box bulk transfer) and `ambiref` (adjective-qualified referent) entity-tracking scenarios. Raises Entity tracking from **40.42 → 45.89** (+5.47).

### Key Results

| System | BLiMP | EWoK | Entity | AoA | GLUE | TextAvg |
|--------|-------|------|--------|-----|------|---------|
| Q baseline (pre-train only) | 76.00 | 52.68 | 40.42 | 9.05 | — | — |
| + CDI Curriculum + SimPER v1 | 75.03 | 55.33 | 42.73 | 17.24 | 68.91 | 47.73 |
| **+ CDI Curriculum + SimPER v2 (ours)** | **74.99** | **54.70** | **45.89** | **17.24** | **67.72** | **47.80** |
| Leaderboard #1 (prabhasa) | — | — | — | — | — | 47.50 |

---

## Repository Structure

```
.
├── train_hybrid_lm.py          # GPT-BERT pre-training (75% MLM + 25% CLM)
├── build_aoa_curriculum.py     # CDI sentence-level curriculum construction
├── data.py                     # Dataset utilities
├── build_aoa_curriculum.py     # CDI AoA curriculum builder
├── build_compliant_dataset.py  # 100M-word budget-compliant corpus builder
├── tokenize_docs_to_bin.py     # Tokenization pipeline
├── teacher_feedback/
│   └── scripts/
│       ├── train_simpo.py              # SimPER / SimPO post-training
│       ├── generate_entity_pairs.py    # Entity preference pair generation
│       ├── generate_preference_pairs.py # Grammar preference pair generation
│       ├── teacher_label.py            # Qwen3-8B teacher labelling
│       ├── run_s1_full.sh              # Full SimPER v2 training run
│       └── run_simper_v2_watchdog.sh   # Watchdog runner
└── paper/
    ├── main.tex
    └── custom.bib
```

---

## Model Architecture

GPT-BERT hybrid encoder (winning BabyLM 2025 architecture):

| Param | Value |
|-------|-------|
| Layers | 12 |
| Hidden dim | 768 |
| Attention heads | 12 |
| Parameters | ~125M |
| Objective | 75% MLM + 25% CLM |
| Position encoding | Relative (bucket size 32) |
| Optimizer | LAMB |
| Peak LR | 7e-3 |
| Total steps | 16,372 |

---

## Training

### 1. Pre-training with CDI Curriculum

```bash
conda activate babyllava

# Build CDI-ordered curriculum dataset
python build_aoa_curriculum.py \
    --corpus babylm_100M_PURE_TEXT_FINAL.json \
    --aoa_percentile 0.10 \
    --phase1_ratio 0.10 \
    --output babylm_curriculum.json

# Pre-train GPT-BERT hybrid
python train_hybrid_lm.py \
    --data babylm_curriculum.json \
    --mlm_ratio 0.75 \
    --steps 16372 \
    --batch 1024 \
    --lr 7e-3 \
    --seq_curriculum 128:0.8,256:1.0
```

### 2. SimPER v2 Post-training

```bash
# Generate entity preference pairs
python teacher_feedback/scripts/generate_entity_pairs.py \
    --scenarios regular move_contents ambiref \
    --n_per_type 5000 \
    --output teacher_feedback/data/entity_pairs.jsonl

# Generate grammar preference pairs
python teacher_feedback/scripts/generate_preference_pairs.py \
    --output teacher_feedback/data/grammar_pairs.jsonl

# (Optional) Teacher labelling with Qwen3-8B
python teacher_feedback/scripts/teacher_label.py \
    --data teacher_feedback/data/abl_A0_combined.jsonl \
    --model Qwen/Qwen3-8B

# Run SimPER post-training (300 steps, lr=1e-5, batch=32)
bash teacher_feedback/scripts/run_s1_full.sh
```

---

## Data Budget

| Phase | Source | Words |
|-------|--------|-------|
| Pre-training | Official BabyLM 2026 Strict (CHILDES, Gutenberg, OpenSubtitles, BNC-Spoken, Switchboard) + Supplementary (Gutenberg extra + Wikipedia 2021) | **100M** |
| Post-training | SimPER entity pairs (novel text only, ~4,800 pairs × 142 w/pair) | **~0.68M** |
| **Total unique text exposure** | | **~100.68M** (<1% over nominal cap) |

---

## Checkpoints

| Checkpoint | Description | Link |
|-----------|-------------|------|
| Final model (CDI + SimPER v2, step 300) | Submitted model, TextAvg 47.80 | [HuggingFace](https://huggingface.co/AliceAndNoob/babylm2026-strict-gptbert-simper) |
| AoA milestone checkpoints (28×) | Steps 122–16372, used for AoA curve evaluation | Included in HF repo |

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("AliceAndNoob/babylm2026-strict-gptbert-simper")
tokenizer = AutoTokenizer.from_pretrained("AliceAndNoob/babylm2026-strict-gptbert-simper")
```

---

## Requirements

```
torch>=2.0
transformers>=4.40
datasets
huggingface_hub
numpy
tqdm
```

Install:
```bash
conda create -n babyllava python=3.10
conda activate babyllava
pip install torch transformers datasets huggingface_hub numpy tqdm
```

---

## Citation

```bibtex
@inproceedings{wang2026cradle,
  title     = {{CRADLE}: Curriculum and Reasoning via Age-of-acquisition-Driven
               Lexical Enhancement for {BabyLM} 2026},
  author    = {Wang, Gan},
  booktitle = {Proceedings of the BabyLM Challenge at CoNLL 2026},
  year      = {2026},
}
```

---

## License

Code: MIT License  
Data: See individual dataset licenses (CHILDES, Gutenberg, OpenSubtitles, BNC, Switchboard, Wikipedia).
