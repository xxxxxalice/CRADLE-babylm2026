#!/bin/bash
# SimPER v2 完整流水线 watchdog
# 功能: 等待标注完成 → 合并 → 训练 → 崩溃自动重启 → 连续失败自动诊断
set -euo pipefail

PYTHON="/home/language/miniconda3/envs/babyllava/bin/python"
DATA="/data0/lexi/babyllava/teacher_feedback/data"
CKPT_BASE="/data0/lexi/babyllava/ckpt_q_seqcurr/hf_q_seqcurr_75mlm_step11787"
OUT_DIR="/data0/lexi/babyllava/teacher_feedback/checkpoints/simper_Q_v2"
TRAIN_LOG="$OUT_DIR/train.log"
WD_LOG="$OUT_DIR/watchdog.log"
SIMPER="/data0/lexi/babyllava/teacher_feedback/scripts/train_simpo.py"
MERGED="$DATA/teacher_labeled_8b.jsonl"
MAX_ATTEMPTS=5
STALE_SECS=300   # 5分钟无日志更新 → 判定卡死

mkdir -p "$OUT_DIR"

wlog() { echo "[watchdog $(date '+%H:%M:%S')] $*" | tee -a "$WD_LOG"; }

# ─── Phase 0: 等待所有标注shard完成 ────────────────────────────────────────────
wlog "Phase 0: 检查标注shard..."
SHARD_LOGS=(
    "$DATA/teacher_label_8b_shard0.log"
    "$DATA/teacher_label_8b_shard1.log"
    "$DATA/teacher_label_8b_shard2_0.log"
    "$DATA/teacher_label_8b_shard2_1.log"
    "$DATA/teacher_label_8b_shard2_2.log"
)
while true; do
    all_done=true
    for f in "${SHARD_LOGS[@]}"; do
        if ! grep -q "标注完成\|=== 标注完成" "$f" 2>/dev/null; then
            last=$(grep "\[" "$f" 2>/dev/null | tail -1 || echo "(未开始)")
            wlog "  等待 $(basename $f): $last"
            all_done=false
            break
        fi
    done
    $all_done && break
    sleep 30
done
wlog "Phase 0: 所有shard标注完成"

# ─── Phase 1: 合并数据 ──────────────────────────────────────────────────────────
if [ ! -f "$MERGED" ] || [ "$(wc -l < "$MERGED")" -lt 1000 ]; then
    wlog "Phase 1: 合并标注数据..."
    $PYTHON /tmp/merge_teacher_labels.py 2>&1 | tee -a "$WD_LOG"
    N=$(wc -l < "$MERGED")
    wlog "Phase 1: 合并完成，共 $N 行"
else
    wlog "Phase 1: 已有合并数据 $(wc -l < "$MERGED") 行，跳过"
fi

# ─── Phase 2: 训练 with watchdog ────────────────────────────────────────────────
# 当前参数配置
LR=1e-5
STEPS=300
BATCH=32
ANCHOR=0.05
WARMUP=30

wlog "Phase 2: 开始 SimPER v2 训练 (lr=$LR, steps=$STEPS, batch=$BATCH)"

attempt=0
while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    wlog "=== 尝试 $attempt/$MAX_ATTEMPTS ==="

    # 找最新checkpoint以断点续传
    LATEST_CKPT=$(ls "$OUT_DIR"/simper_step*/pytorch_model.bin \
                     "$OUT_DIR"/simper_step*/model.safetensors 2>/dev/null \
                  | sed 's|/[^/]*$||' | sort -t_ -k3 -n | tail -1 || true)
    if [ -n "$LATEST_CKPT" ]; then
        STEP_DONE=$(echo "$LATEST_CKPT" | grep -oP 'step\K[0-9]+' || echo 0)
        if [ "${STEP_DONE:-0}" -ge "$STEPS" ]; then
            wlog "已完成 $STEP_DONE/$STEPS 步，训练结束"
            break
        fi
        wlog "从 step$STEP_DONE 断点续训"
        RESUME_ARG="--resume_from $LATEST_CKPT"
    else
        wlog "从头开始训练"
        RESUME_ARG=""
    fi

    # 启动训练
    /home/language/miniconda3/envs/babyllava/bin/torchrun \
        --standalone --nproc_per_node=4 \
        "$SIMPER" \
        --model_path "$CKPT_BASE" \
        --data_path "$MERGED" \
        --output_dir "$OUT_DIR" \
        --loss_type simper \
        --lr "$LR" \
        --batch_size "$BATCH" \
        --max_steps "$STEPS" \
        --max_length 128 \
        --save_every 100 \
        --log_every 10 \
        --warmup_steps "$WARMUP" \
        --anchor_weight "$ANCHOR" \
        --teacher_only \
        $RESUME_ARG \
        >> "$TRAIN_LOG" 2>&1 &
    TRAIN_PID=$!
    wlog "训练进程 PID: $TRAIN_PID"

    # 监控：检测卡死 + 自然结束
    while kill -0 $TRAIN_PID 2>/dev/null; do
        sleep 60
        ! kill -0 $TRAIN_PID 2>/dev/null && break

        # 检测日志是否停止更新（卡死检测）
        if [ -f "$TRAIN_LOG" ]; then
            AGE=$(( $(date +%s) - $(stat -c %Y "$TRAIN_LOG" 2>/dev/null || echo 0) ))
            if [ $AGE -gt $STALE_SECS ]; then
                wlog "警告: 日志 ${AGE}s 未更新，判定卡死，终止进程"
                kill $TRAIN_PID 2>/dev/null || true
                sleep 5
                break
            fi
        fi
    done

    wait $TRAIN_PID 2>/dev/null
    EXIT=$?
    wlog "退出码: $EXIT"

    # 检查是否真正完成
    LATEST_CKPT=$(ls "$OUT_DIR"/simper_step*/pytorch_model.bin \
                     "$OUT_DIR"/simper_step*/model.safetensors 2>/dev/null \
                  | sed 's|/[^/]*$||' | sort -t_ -k3 -n | tail -1 || true)
    STEP_DONE=$(echo "${LATEST_CKPT}" | grep -oP 'step\K[0-9]+' 2>/dev/null || echo 0)

    if [ "$EXIT" -eq 0 ] || [ "${STEP_DONE:-0}" -ge "$STEPS" ]; then
        wlog "训练成功完成 (step=${STEP_DONE})"
        break
    fi

    # ─ 自动诊断 ──────────────────────────────────────────────────────────────────
    wlog "=== 自动诊断 (尝试 $attempt) ==="
    DEBUG_FILE="$OUT_DIR/debug_attempt${attempt}.txt"
    {
        echo "=== 诊断报告 attempt=$attempt $(date) ==="
        echo "退出码: $EXIT"
        echo ""
        echo "--- 训练日志尾部 (最后50行) ---"
        tail -50 "$TRAIN_LOG" 2>/dev/null || echo "(无日志)"
        echo ""
        echo "--- GPU状态 ---"
        nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null
        echo ""
        echo "--- 常见错误检测 ---"
    } > "$DEBUG_FILE"

    # OOM检测
    if grep -qiE "out of memory|CUDA error.*alloc|OOM" "$TRAIN_LOG" 2>/dev/null; then
        wlog "检测到 CUDA OOM，将 batch_size $BATCH → $((BATCH/2))"
        echo "OOM: batch $BATCH → $((BATCH/2))" >> "$DEBUG_FILE"
        BATCH=$((BATCH/2))
    # 梯度/数值问题
    elif grep -qiE "nan|inf.*loss|loss.*nan|gradient.*norm" "$TRAIN_LOG" 2>/dev/null; then
        wlog "检测到数值不稳定，将 lr $LR → $(echo "$LR * 0.1" | bc -l | head -c 8)"
        OLD_LR=$LR
        LR=$(python3 -c "print(f'{float('$LR')*0.1:.2e}')")
        echo "数值不稳定: lr $OLD_LR → $LR" >> "$DEBUG_FILE"
    # inplace操作
    elif grep -qiE "inplace.*operation|version.*mismatch" "$TRAIN_LOG" 2>/dev/null; then
        wlog "检测到 inplace 操作错误（model_extra.py bug）"
        echo "inplace op: 检查 model_extra.py 修复" >> "$DEBUG_FILE"
    # broadcast_buffers
    elif grep -qiE "broadcast_buffers|EmbeddingBackward" "$TRAIN_LOG" 2>/dev/null; then
        wlog "检测到 DDP broadcast_buffers 问题（已修复，不应再出现）"
        echo "DDP: broadcast_buffers 问题" >> "$DEBUG_FILE"
    else
        wlog "未知错误，保守等待 30s 后重试"
        echo "未知: 见日志尾部" >> "$DEBUG_FILE"
    fi

    wlog "诊断报告: $DEBUG_FILE"
    sleep 30
done

# ─── 最终结果 ─────────────────────────────────────────────────────────────────
if [ $attempt -ge $MAX_ATTEMPTS ]; then
    wlog "❌ 连续 $MAX_ATTEMPTS 次失败，停止重试"
    wlog "调试信息汇总: ls $OUT_DIR/debug_attempt*.txt"
    exit 1
fi

wlog "✓ SimPER v2 训练完成"
wlog "checkpoint: $OUT_DIR/"
wlog "下一步: 运行全评测"
