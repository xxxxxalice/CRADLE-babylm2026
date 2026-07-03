#!/bin/bash
# SimPER/SimPO 完整流水线 (基于2024-2025年最优实践):
#
#   Step 1: 生成原始偏好对 (语法扰动 + 实体追踪, 带margin质量过滤)
#   Step 2: Qwen3-VL-4B 双向swap教师标注 (消除位置偏差)
#   Step 3: SimPER 微调 (4卡, 无超参, NeurIPS 2025)
#            备选: SimPO (β=2.5, γ=1.0) 或 KTO (非成对数据)
#
# 方法选择依据 (arXiv:2502.00883 vs arXiv:2405.14734):
#   SimPER > SimPO:  +4.9~5.2分AlpacaEval2，无需调β/γ
#   SimPO β须2.0~3.0: 原始代码用0.1是错的 (差25倍信号强度!)
#
# 数据量依据 (arXiv:2502.14560 "Less is More"):
#   500~2000高质量对 > 60000弱质量对
#   双向swap一致率约50~70%，20K原始对 → ~5K高质量训练对
#
# 用法:
#   bash run_simpo.sh [HF_CKPT_PATH] [LOSS_TYPE]
#   LOSS_TYPE: simper(默认) / simpo / kto

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$BASE_DIR/data"
CKPT_DIR="$BASE_DIR/checkpoints"
LOG_DIR="$BASE_DIR/logs"
PYTHON=/home/language/miniconda3/envs/babyllava/bin/python

mkdir -p "$DATA_DIR" "$CKPT_DIR" "$LOG_DIR"

DEFAULT_CKPT="/data0/lexi/babyllava/ckpt_s_infomask/hf_s_infomask_75mlm_step11787"
MODEL_PATH="${1:-$DEFAULT_CKPT}"
LOSS_TYPE="${2:-simper}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/simpo_pipeline.log"; }

log "=== 偏好优化流水线 (方法: $LOSS_TYPE) ==="
log "基础模型: $MODEL_PATH"

# ════════════════════════════════════════════════════════════════════════
# Step 1: 生成偏好对 (带margin质量过滤)
# ════════════════════════════════════════════════════════════════════════
if [ -f "$DATA_DIR/all_pairs_raw.jsonl" ]; then
    N=$(wc -l < "$DATA_DIR/all_pairs_raw.jsonl")
    log "跳过生成: all_pairs_raw.jsonl 已存在 ($N 行)"
else
    log "=== Step 1: 生成偏好对 (目标: ~20K, 含margin过滤) ==="
    $PYTHON "$SCRIPT_DIR/generate_preference_pairs.py" \
        --n_grammar 15000 \
        --n_entity   5000 \
        --seed 42 \
        2>&1 | tee -a "$LOG_DIR/step1_generate.log"
    log "Step 1 完成"
fi

# ════════════════════════════════════════════════════════════════════════
# Step 2: 双向swap教师标注
# ════════════════════════════════════════════════════════════════════════
if [ -f "$DATA_DIR/teacher_labeled.jsonl" ]; then
    N=$(wc -l < "$DATA_DIR/teacher_labeled.jsonl")
    log "跳过标注: teacher_labeled.jsonl 已存在 ($N 行)"
else
    log "=== Step 2: Qwen3-VL-4B 双向swap标注 (GPU 0) ==="
    log "注意: 双向swap每对需2次推理，约需2~3小时"
    CUDA_VISIBLE_DEVICES=0 $PYTHON "$SCRIPT_DIR/teacher_label.py" \
        --input       "$DATA_DIR/all_pairs_raw.jsonl" \
        --output      "$DATA_DIR/teacher_labeled.jsonl" \
        --device      cuda:0 \
        --certain_only \
        2>&1 | tee -a "$LOG_DIR/step2_teacher.log"
    log "Step 2 完成"

    # 统计有效训练数据量
    N_GOOD=$(wc -l < "$DATA_DIR/teacher_labeled.jsonl")
    log "有效训练对 (双向一致): $N_GOOD"
    if [ "${N_GOOD:-0}" -lt 200 ]; then
        log "警告: 有效对不足200，考虑去掉 --certain_only 或增大 --n_grammar"
    fi
fi

# ════════════════════════════════════════════════════════════════════════
# Step 3: 偏好优化微调 (4卡)
# ════════════════════════════════════════════════════════════════════════
log "=== Step 3: $LOSS_TYPE 微调 (4卡) ==="

if [ ! -d "$MODEL_PATH" ]; then
    log "错误: 基础模型不存在: $MODEL_PATH"
    log "请确认S组训练已完成并生成HF格式checkpoint"
    exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# SimPER默认超参: 无β/γ，lr=5e-5，anchor_weight=0.1
# SimPO超参 (文献最优): β=2.5, γ=1.0 (γ/β≈0.4)
if [ "$LOSS_TYPE" = "simpo" ]; then
    EXTRA_ARGS="--beta 2.5 --gamma_margin 1.0"
elif [ "$LOSS_TYPE" = "kto" ]; then
    EXTRA_ARGS="--beta 0.1"
else
    EXTRA_ARGS=""
fi

/home/language/miniconda3/envs/babyllava/bin/torchrun \
    --standalone --nproc_per_node=4 \
    "$SCRIPT_DIR/train_simpo.py" \
    --model_path    "$MODEL_PATH" \
    --data_path     "$DATA_DIR/teacher_labeled.jsonl" \
    --output_dir    "$CKPT_DIR/${LOSS_TYPE}_v1" \
    --loss_type     "$LOSS_TYPE" \
    --lr            5e-5 \
    --batch_size    64 \
    --max_steps     1000 \
    --max_length    128 \
    --save_every    250 \
    --log_every     25 \
    --warmup_steps  50 \
    --anchor_weight 0.1 \
    --teacher_only  \
    $EXTRA_ARGS \
    2>&1 | tee -a "$LOG_DIR/step3_${LOSS_TYPE}.log"

log "=== 流水线完成 ==="
log "最终checkpoint: $CKPT_DIR/${LOSS_TYPE}_v1/${LOSS_TYPE}_step1000_final"
log ""
log "下一步: 评测"
log "  bash /data0/lexi/babyllava/gpt-bert/sweep_and_fulleval_s.sh \\"
log "    [将CKPT_DIR改为 $CKPT_DIR/${LOSS_TYPE}_v1/]"
