#!/bin/bash
# 消融实验并行 pipeline: 4 卡同时跑不同实验
# Batch1: A0/A1/B1/B2 → 4路评测
# Batch2: C1/C2/C3/D1 → 4路评测
# 最后打印汇总表
set -euo pipefail

PYTHON=/home/language/miniconda3/envs/babyllava/bin/python
TORCHRUN=/home/language/miniconda3/envs/babyllava/bin/torchrun
SIMPER=/data0/lexi/babyllava/teacher_feedback/scripts/train_simpo.py
DATA=/data0/lexi/babyllava/teacher_feedback/data
CKPT_Q=/data0/lexi/babyllava/ckpt_q_seqcurr/hf_q_seqcurr_75mlm_step11787
OUT_BASE=/data0/lexi/babyllava/teacher_feedback/checkpoints
EVAL_DIR=/data0/lexi/babyllava/babylm-eval/strict
EVAL_DATA=$EVAL_DIR/evaluation_data/full_eval
LOG_DIR=/data0/lexi/babyllava/fulleval_logs
MERGED=$DATA/teacher_labeled_8b.jsonl

mkdir -p "$LOG_DIR"
log() { echo "[abl $(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/ablation_parallel.log"; }

# ── 单卡训练函数 ────────────────────────────────────────────────────────────────
# 全局 batch=32，单卡 per-device=32 ≡ 4卡 per-device=8，等效
train_1gpu() {
    local gpu=$1 name=$2 data=$3 lr=$4 steps=$5
    local loss=${6:-simper} batch=${7:-32} anchor=${8:-0.05}
    local out="$OUT_BASE/$name"
    mkdir -p "$out"

    # 断点跳过
    local final="$out/${loss}_step${steps}_final"
    if [ -d "$final" ] && { [ -f "$final/pytorch_model.bin" ] || [ -f "$final/model.safetensors" ]; }; then
        log "[$name] 已完成，跳过"
        return 0
    fi

    log "[$name] GPU$gpu 开始 (lr=$lr, steps=$steps, loss=$loss, data=$(basename $data))"
    CUDA_VISIBLE_DEVICES=$gpu \
    $TORCHRUN --standalone --nproc_per_node=1 \
        --master_port $((29500 + gpu)) \
        "$SIMPER" \
        --model_path  "$CKPT_Q" \
        --data_path   "$data" \
        --output_dir  "$out" \
        --loss_type   "$loss" \
        --lr          "$lr" \
        --batch_size  "$batch" \
        --max_steps   "$steps" \
        --max_length  128 \
        --save_every  100 \
        --log_every   20 \
        --warmup_steps 30 \
        --anchor_weight "$anchor" \
        --teacher_only \
        >> "$out/train.log" 2>&1
    log "[$name] GPU$gpu 训练完成"
}

# ── 单卡评测函数 ────────────────────────────────────────────────────────────────
eval_1gpu() {
    local gpu=$1 ckpt=$2 tag=$3
    local log_file="$LOG_DIR/${tag}.log"
    log "[$tag] GPU$gpu 评测开始"
    export CUDA_VISIBLE_DEVICES=$gpu TOKENIZERS_PARALLELISM=false
    cd "$EVAL_DIR"
    {
    echo "=== $tag ==="
    echo "[BLiMP]"
    $PYTHON -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$ckpt" --backend mntp \
        --task blimp --data_path "$EVAL_DATA/blimp_filtered" \
        --save_predictions 2>/dev/null | grep -A1 "^### AVERAGE" | tail -1
    echo "[BLiMP-supplement]"
    $PYTHON -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$ckpt" --backend mntp \
        --task blimp --data_path "$EVAL_DATA/supplement_filtered" \
        --save_predictions 2>/dev/null | grep -A1 "^### AVERAGE" | tail -1
    echo "[EWoK]"
    $PYTHON -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$ckpt" --backend mntp \
        --task ewok --data_path "$EVAL_DATA/ewok_filtered" \
        --save_predictions 2>/dev/null | grep -A1 "^### AVERAGE" | tail -1
    echo "[Entity-tracking]"
    $PYTHON -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$ckpt" --backend mntp \
        --task entity_tracking --data_path "$EVAL_DATA/entity_tracking" \
        --save_predictions 2>/dev/null | grep -A1 "^### AVERAGE" | tail -1
    echo "[COMPS]"
    $PYTHON -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$ckpt" --backend mntp \
        --task comps --data_path "$EVAL_DATA/comps" \
        --save_predictions 2>/dev/null | grep -A1 "^### AVERAGE" | tail -1
    echo "[Reading]"
    $PYTHON -m evaluation_pipeline.reading.run \
        --model_path_or_name "$ckpt" --backend mntp \
        --data_path "$EVAL_DATA/reading/reading_data.csv" 2>/dev/null \
    | grep -iE "EYE TRACKING SCORE|SELF-PACED READING SCORE"
    echo "=== $tag DONE ==="
    } 2>&1 | tee "$log_file"
    log "[$tag] GPU$gpu 评测完成"
}

find_ckpt() {
    local out=$1 loss=${2:-simper}
    ls "$out/${loss}_step"*_final/pytorch_model.bin \
       "$out/${loss}_step"*_final/model.safetensors 2>/dev/null \
    | sed 's|/[^/]*$||' | sort -t_ -k3 -n | tail -1 || true
}

# ════════════════════════════════════════════════════════════════════════════════
# Phase 0: 评测已完成的 P1
# ════════════════════════════════════════════════════════════════════════════════
log "=== Phase 0: 评测 P1 (SimPER v2 on R) ==="
P1_CKPT="$OUT_BASE/simper_R_v2/simper_step300_final"
eval_1gpu 0 "$P1_CKPT" "SimPER_R_v2"

# ════════════════════════════════════════════════════════════════════════════════
# Batch 1: 论点2 + 论点4消融 (GPU0-3同时)
# ════════════════════════════════════════════════════════════════════════════════
log "=== Batch1 训练: A0/A1/B1/B2 四路并行 ==="
train_1gpu 0 "abl_A0_nofilter" "$DATA/abl_A0_nofilter.jsonl"     "1e-5" 300 &
train_1gpu 1 "abl_A1_certain"  "$DATA/abl_A1_certain.jsonl"      "1e-5" 300 &
train_1gpu 2 "abl_B1_entity"   "$DATA/abl_B1_entity_only.jsonl"  "1e-5" 300 &
train_1gpu 3 "abl_B2_grammar"  "$DATA/abl_B2_grammar_only.jsonl" "1e-5" 300 &
wait
log "=== Batch1 训练完成 ==="

log "=== Batch1 评测: 四路并行 ==="
eval_1gpu 0 "$(find_ckpt $OUT_BASE/abl_A0_nofilter)" "Abl_A0_nofilter"  &
eval_1gpu 1 "$(find_ckpt $OUT_BASE/abl_A1_certain)"  "Abl_A1_certain"   &
eval_1gpu 2 "$(find_ckpt $OUT_BASE/abl_B1_entity)"   "Abl_B1_entity_only"  &
eval_1gpu 3 "$(find_ckpt $OUT_BASE/abl_B2_grammar)"  "Abl_B2_grammar_only" &
wait
log "=== Batch1 评测完成 ==="

# ════════════════════════════════════════════════════════════════════════════════
# Batch 2: 论点3消融 (GPU0-3同时)
# ════════════════════════════════════════════════════════════════════════════════
log "=== Batch2 训练: C1/C2/C3/D1 四路并行 ==="
train_1gpu 0 "abl_C1_step100" "$MERGED" "1e-5" 100           &
train_1gpu 1 "abl_C2_step500" "$MERGED" "1e-5" 500           &
train_1gpu 2 "abl_C3_lr2e5"   "$MERGED" "2e-5" 300           &
train_1gpu 3 "abl_D1_simpo"   "$MERGED" "1e-5" 300 "simpo"   &
wait
log "=== Batch2 训练完成 ==="

log "=== Batch2 评测: 四路并行 ==="
eval_1gpu 0 "$(find_ckpt $OUT_BASE/abl_C1_step100)" "Abl_C1_step100" &
eval_1gpu 1 "$(find_ckpt $OUT_BASE/abl_C2_step500)" "Abl_C2_step500" &
eval_1gpu 2 "$(find_ckpt $OUT_BASE/abl_C3_lr2e5)"  "Abl_C3_lr2e5"   &
eval_1gpu 3 "$(find_ckpt "$OUT_BASE/abl_D1_simpo" "simpo")" "Abl_D1_simpo" &
wait
log "=== Batch2 评测完成 ==="

# ════════════════════════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════════════════════════
log "=== 全部完成，汇总结果 ==="
$PYTHON - <<'PYEOF'
import re

def parse(path):
    try:
        content = open(path).read()
    except:
        return {}
    r = {}
    for key, pat in [
        ('BLiMP',   r'\[BLiMP\].*?AVERAGE ACCURACY\s+([\d.]+)'),
        ('BLiMP-S', r'\[BLiMP-supplement\].*?AVERAGE ACCURACY\s+([\d.]+)'),
        ('EWoK',    r'\[EWoK\].*?AVERAGE ACCURACY\s+([\d.]+)'),
        ('Entity',  r'\[Entity-tracking\].*?AVERAGE ACCURACY\s+([\d.]+)'),
        ('COMPS',   r'\[COMPS\].*?AVERAGE ACCURACY\s+([\d.]+)'),
        ('Eye',     r'EYE TRACKING SCORE:\s+([\d.]+)'),
        ('SPR',     r'SELF-PACED READING SCORE:\s+([\d.]+)'),
    ]:
        m = re.search(pat, content, re.DOTALL)
        if m: r[key] = float(m.group(1))
    return r

LOG = '/data0/lexi/babyllava/fulleval_logs'
exps = [
    ('Q基线(参考)',         'Q_seqcurr_step11787'),
    ('SimPERv2/Q(参考)',   'simper_Q_v2_fulleval'),
    ('P1: R+SimPERv2',    'SimPER_R_v2'),
    ('─论点2─',''),
    ('A0 无teacher过滤',   'Abl_A0_nofilter'),
    ('A1 certain+反转',    'Abl_A1_certain'),
    ('─论点4─',''),
    ('B1 entity-only',    'Abl_B1_entity_only'),
    ('B2 grammar-only',   'Abl_B2_grammar_only'),
    ('─论点3 步数─',''),
    ('C1 100步',           'Abl_C1_step100'),
    ('C2 500步',           'Abl_C2_step500'),
    ('─论点3 lr─',''),
    ('C3 lr=2e-5',         'Abl_C3_lr2e5'),
    ('─论点3 loss─',''),
    ('D1 SimPO',           'Abl_D1_simpo'),
]

hdr = f"{'实验':22s} | {'BLiMP':6s} | {'BLiMP-S':7s} | {'EWoK':6s} | {'Entity':6s} | {'COMPS':6s} | Eye  | SPR"
print('\n' + hdr)
print('-' * 82)
for label, tag in exps:
    if not tag:
        print(f"  {'─'*22}  {label}")
        continue
    r = parse(f'{LOG}/{tag}.log')
    def f(k): return f"{r[k]:.2f}" if k in r else '  —  '
    print(f"{label:22s} | {f('BLiMP'):6s} | {f('BLiMP-S'):7s} | {f('EWoK'):6s} | {f('Entity'):6s} | {f('COMPS'):6s} | {f('Eye'):4s} | {f('SPR'):4s}")
PYEOF
