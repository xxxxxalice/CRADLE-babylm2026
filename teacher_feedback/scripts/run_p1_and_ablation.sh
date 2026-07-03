#!/bin/bash
# P1 + 消融实验完整 pipeline
# P1: SimPER v2 on R(step11500)
# 消融: 8组 × Q基线, 覆盖论点2/3/4
set -euo pipefail

PYTHON=/home/language/miniconda3/envs/babyllava/bin/python
TORCHRUN=/home/language/miniconda3/envs/babyllava/bin/torchrun
SIMPER=/data0/lexi/babyllava/teacher_feedback/scripts/train_simpo.py
DATA=/data0/lexi/babyllava/teacher_feedback/data
CKPT_Q=/data0/lexi/babyllava/ckpt_q_seqcurr/hf_q_seqcurr_75mlm_step11787
CKPT_R=/data0/lexi/babyllava/ckpt_r_3phase/hf_r_3phase_75mlm_step11500
OUT_BASE=/data0/lexi/babyllava/teacher_feedback/checkpoints
EVAL_DIR=/data0/lexi/babyllava/babylm-eval/strict
EVAL_DATA=$EVAL_DIR/evaluation_data/full_eval
LOG_DIR=/data0/lexi/babyllava/fulleval_logs
MERGED=$DATA/teacher_labeled_8b.jsonl

mkdir -p "$LOG_DIR"

log() { echo "[pipeline $(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/p1_ablation_pipeline.log"; }

# ── 训练函数 ────────────────────────────────────────────────────────────────────
train_simper() {
    local name=$1       # 实验名, 用于目录和日志
    local base_ckpt=$2  # 基础模型路径
    local data_file=$3  # 训练数据
    local lr=$4
    local steps=$5
    local loss=${6:-simper}
    local batch=${7:-32}
    local anchor=${8:-0.05}

    local out_dir="$OUT_BASE/$name"
    local train_log="$out_dir/train.log"
    mkdir -p "$out_dir"

    log "=== 开始训练 $name (lr=$lr, steps=$steps, loss=$loss, data=$(basename $data_file)) ==="

    # 断点续训: 找最新 checkpoint
    local resume_arg=""
    local latest=$(ls "$out_dir"/${loss}_step*/pytorch_model.bin \
                      "$out_dir"/${loss}_step*/model.safetensors 2>/dev/null \
                   | sed 's|/[^/]*$||' | sort -t_ -k3 -n | tail -1 || true)
    if [ -n "$latest" ]; then
        local done_step=$(echo "$latest" | grep -oP 'step\K[0-9]+' || echo 0)
        if [ "${done_step:-0}" -ge "$steps" ]; then
            log "$name 已完成 (step$done_step), 跳过训练"
            return 0
        fi
        resume_arg="--resume_from $latest"
        log "$name 从 step$done_step 断点续训"
    fi

    $TORCHRUN --standalone --nproc_per_node=4 "$SIMPER" \
        --model_path  "$base_ckpt" \
        --data_path   "$data_file" \
        --output_dir  "$out_dir" \
        --loss_type   "$loss" \
        --lr          "$lr" \
        --batch_size  "$batch" \
        --max_steps   "$steps" \
        --max_length  128 \
        --save_every  100 \
        --log_every   10 \
        --warmup_steps 30 \
        --anchor_weight "$anchor" \
        --teacher_only \
        $resume_arg \
        >> "$train_log" 2>&1

    log "$name 训练完成"
}

# ── 评测函数 ────────────────────────────────────────────────────────────────────
run_fulleval() {
    local ckpt_path=$1  # HF checkpoint 目录
    local tag=$2        # 日志标签
    local gpu=${3:-0}

    local log="$LOG_DIR/${tag}.log"
    log "=== 评测 $tag (GPU$gpu) ==="

    export CUDA_VISIBLE_DEVICES=$gpu
    export TOKENIZERS_PARALLELISM=false
    cd "$EVAL_DIR"

    {
    echo "=== $tag ==="

    echo "[BLiMP]"
    $PYTHON -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$ckpt_path" --backend mntp \
        --task blimp --data_path "$EVAL_DATA/blimp_filtered" \
        --save_predictions 2>/dev/null \
    | grep -A1 "^### AVERAGE" | tail -1

    echo "[BLiMP-supplement]"
    $PYTHON -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$ckpt_path" --backend mntp \
        --task blimp --data_path "$EVAL_DATA/supplement_filtered" \
        --save_predictions 2>/dev/null \
    | grep -A1 "^### AVERAGE" | tail -1

    echo "[EWoK]"
    $PYTHON -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$ckpt_path" --backend mntp \
        --task ewok --data_path "$EVAL_DATA/ewok_filtered" \
        --save_predictions 2>/dev/null \
    | grep -A1 "^### AVERAGE" | tail -1

    echo "[Entity-tracking]"
    $PYTHON -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$ckpt_path" --backend mntp \
        --task entity_tracking --data_path "$EVAL_DATA/entity_tracking" \
        --save_predictions 2>/dev/null \
    | grep -A1 "^### AVERAGE" | tail -1

    echo "[COMPS]"
    $PYTHON -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$ckpt_path" --backend mntp \
        --task comps --data_path "$EVAL_DATA/comps" \
        --save_predictions 2>/dev/null \
    | grep -A1 "^### AVERAGE" | tail -1

    echo "[Reading]"
    $PYTHON -m evaluation_pipeline.reading.run \
        --model_path_or_name "$ckpt_path" --backend mntp \
        --data_path "$EVAL_DATA/reading/reading_data.csv" 2>/dev/null \
    | grep -iE "EYE TRACKING SCORE|SELF-PACED READING SCORE"

    echo "=== $tag DONE ==="
    } 2>&1 | tee "$log"
}

find_final_ckpt() {
    local out_dir=$1
    local loss=${2:-simper}
    ls "$out_dir"/${loss}_step*_final/pytorch_model.bin \
       "$out_dir"/${loss}_step*_final/model.safetensors 2>/dev/null \
    | sed 's|/[^/]*$||' | sort -t_ -k3 -n | tail -1 || true
}

# ════════════════════════════════════════════════════════════════════════════════
# P1: SimPER v2 on R(step11500)
# ════════════════════════════════════════════════════════════════════════════════
log "========== P1: SimPER v2 on R(step11500) =========="
train_simper "simper_R_v2" "$CKPT_R" "$MERGED" "1e-5" 300

ckpt_p1=$(find_final_ckpt "$OUT_BASE/simper_R_v2")
if [ -n "$ckpt_p1" ]; then
    run_fulleval "$ckpt_p1" "SimPER_R_v2"
else
    log "P1: 未找到 final checkpoint, 跳过评测"
fi

# ════════════════════════════════════════════════════════════════════════════════
# 论点2消融: Teacher 标注的必要性
# ════════════════════════════════════════════════════════════════════════════════
log "========== 论点2消融: Teacher标注必要性 =========="

# Abl-A0: 不过滤, 全部原始对 (20000)
train_simper "abl_A0_nofilter" "$CKPT_Q" "$DATA/abl_A0_nofilter.jsonl" "1e-5" 300
ckpt=$(find_final_ckpt "$OUT_BASE/abl_A0_nofilter")
[ -n "$ckpt" ] && run_fulleval "$ckpt" "Abl_A0_nofilter"

# Abl-A1: teacher_certain=True (7980对, 含split双向一致)
train_simper "abl_A1_certain" "$CKPT_Q" "$DATA/abl_A1_certain.jsonl" "1e-5" 300
ckpt=$(find_final_ckpt "$OUT_BASE/abl_A1_certain")
[ -n "$ckpt" ] && run_fulleval "$ckpt" "Abl_A1_certain"

# ════════════════════════════════════════════════════════════════════════════════
# 论点4消融: 哪种数据提升 Entity-tracking
# ════════════════════════════════════════════════════════════════════════════════
log "========== 论点4消融: 数据类型 =========="

# Abl-B1: entity-only (2668对)
train_simper "abl_B1_entity" "$CKPT_Q" "$DATA/abl_B1_entity_only.jsonl" "1e-5" 300
ckpt=$(find_final_ckpt "$OUT_BASE/abl_B1_entity")
[ -n "$ckpt" ] && run_fulleval "$ckpt" "Abl_B1_entity_only"

# Abl-B2: grammar-only (4715对)
train_simper "abl_B2_grammar" "$CKPT_Q" "$DATA/abl_B2_grammar_only.jsonl" "1e-5" 300
ckpt=$(find_final_ckpt "$OUT_BASE/abl_B2_grammar")
[ -n "$ckpt" ] && run_fulleval "$ckpt" "Abl_B2_grammar_only"

# ════════════════════════════════════════════════════════════════════════════════
# 论点3消融: 训练配置 (lr / steps / loss_type)
# ════════════════════════════════════════════════════════════════════════════════
log "========== 论点3消融: 步数 =========="

# Abl-C1: 100步
train_simper "abl_C1_step100" "$CKPT_Q" "$MERGED" "1e-5" 100
ckpt=$(find_final_ckpt "$OUT_BASE/abl_C1_step100")
[ -n "$ckpt" ] && run_fulleval "$ckpt" "Abl_C1_step100"

# Abl-C2: 500步
train_simper "abl_C2_step500" "$CKPT_Q" "$MERGED" "1e-5" 500
ckpt=$(find_final_ckpt "$OUT_BASE/abl_C2_step500")
[ -n "$ckpt" ] && run_fulleval "$ckpt" "Abl_C2_step500"

log "========== 论点3消融: 学习率 =========="

# Abl-C3: lr=2e-5
train_simper "abl_C3_lr2e5" "$CKPT_Q" "$MERGED" "2e-5" 300
ckpt=$(find_final_ckpt "$OUT_BASE/abl_C3_lr2e5")
[ -n "$ckpt" ] && run_fulleval "$ckpt" "Abl_C3_lr2e5"

log "========== 论点3消融: loss函数 =========="

# Abl-D1: SimPO loss
train_simper "abl_D1_simpo" "$CKPT_Q" "$MERGED" "1e-5" 300 "simpo"
ckpt=$(find_final_ckpt "$OUT_BASE/abl_D1_simpo" "simpo")
[ -n "$ckpt" ] && run_fulleval "$ckpt" "Abl_D1_simpo"

# ════════════════════════════════════════════════════════════════════════════════
log "========== 全部实验完成, 汇总结果 =========="

python3 - <<'PYEOF'
import re, os

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
    ('Q基线(参考)',   'Q_seqcurr_step11787'),
    ('SimPER-v2(Q)', 'simper_Q_v2_fulleval'),
    ('P1: R+SimPER', 'SimPER_R_v2'),
    ('Abl-A0 无filter','Abl_A0_nofilter'),
    ('Abl-A1 certain','Abl_A1_certain'),
    ('Abl-B1 entity','Abl_B1_entity_only'),
    ('Abl-B2 grammar','Abl_B2_grammar_only'),
    ('Abl-C1 100步',  'Abl_C1_step100'),
    ('Abl-C2 500步',  'Abl_C2_step500'),
    ('Abl-C3 lr2e-5', 'Abl_C3_lr2e5'),
    ('Abl-D1 SimPO',  'Abl_D1_simpo'),
]

print(f"\n{'实验':20s} | {'BLiMP':6s} | {'BLiMP-S':7s} | {'EWoK':6s} | {'Entity':6s} | {'COMPS':6s} | {'Eye':4s} | {'SPR':4s}")
print('-'*80)
for label, tag in exps:
    r = parse(f'{LOG}/{tag}.log')
    def f(k): return f"{r[k]:.2f}" if k in r else '  —  '
    print(f"{label:20s} | {f('BLiMP'):6s} | {f('BLiMP-S'):7s} | {f('EWoK'):6s} | {f('Entity'):6s} | {f('COMPS'):6s} | {f('Eye'):4s} | {f('SPR'):4s}")
PYEOF

log "pipeline 结束"
