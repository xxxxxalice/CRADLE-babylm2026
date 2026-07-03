#!/bin/bash
# S1 过滤消融（完整数据版）
# 实验1: v2_full   — 11101 对 teacher_correct 过滤
# 实验2: A0_11101k — 11101 对无过滤（控制组，等量）
# 对比已有: SimPER A0 (20000无过滤), SimPER v2 (7383过滤), A0_7k (7383无过滤)

set -e

PYTHON="/home/language/miniconda3/envs/babyllava/bin/python"
TORCHRUN="/home/language/miniconda3/envs/babyllava/bin/torchrun"
DATA="/data0/lexi/babyllava/teacher_feedback/data"
CKPT_BASE="/data0/lexi/babyllava/ckpt_q_seqcurr/hf_q_seqcurr_75mlm_step11787"
SIMPER="/data0/lexi/babyllava/teacher_feedback/scripts/train_simpo.py"
CKPT_DIR="/data0/lexi/babyllava/teacher_feedback/checkpoints"
LOG="/data0/lexi/babyllava/teacher_feedback/s1_full.log"

LR=1e-5
STEPS=300
BATCH=32

# 等 Q_full 训练结束（检测 train_multi_gpu 进程消失）
echo "[S1] 等待 Q_full 训练完成..." | tee -a "$LOG"
until ! pgrep -f "train_multi_gpu.py" > /dev/null 2>&1; do
    sleep 60
done
echo "[S1] Q_full 完成，10s 后开始 SimPER 实验..." | tee -a "$LOG"
sleep 10

run_simper() {
    local name="$1"
    local data_file="$2"
    local teacher_only_flag="$3"   # "--teacher_only" 或 "--no-teacher_only"
    local out_dir="$CKPT_DIR/${name}"

    echo "" | tee -a "$LOG"
    echo "=============================" | tee -a "$LOG"
    echo "[S1] 开始训练: $name" | tee -a "$LOG"
    echo "[S1] 数据: $data_file" | tee -a "$LOG"
    echo "[S1] teacher_only: $teacher_only_flag" | tee -a "$LOG"
    echo "=============================" | tee -a "$LOG"

    mkdir -p "$out_dir"

    for attempt in 1 2 3; do
        # 找最新 checkpoint
        LATEST=$(ls "$out_dir"/simper_step*/pytorch_model.bin \
                    "$out_dir"/simper_step*/model.safetensors 2>/dev/null \
                 | sed 's|/[^/]*$||' \
                 | awk -F'step' '{print $2, $0}' | sort -k1 -n | tail -1 | cut -d' ' -f2- || true)
        if [ -n "$LATEST" ]; then
            STEP_DONE=$(echo "$LATEST" | grep -oP 'step\K[0-9]+' || echo 0)
            if [ "${STEP_DONE:-0}" -ge "$STEPS" ]; then
                echo "[S1] $name 已完成 step $STEP_DONE，跳过" | tee -a "$LOG"
                return 0
            fi
            RESUME="--resume_from $LATEST"
        else
            RESUME=""
        fi

        $TORCHRUN --standalone --nproc_per_node=4 \
            "$SIMPER" \
            --model_path "$CKPT_BASE" \
            --data_path  "$data_file" \
            --output_dir "$out_dir" \
            --loss_type  simper \
            --lr         "$LR" \
            --batch_size "$BATCH" \
            --max_steps  "$STEPS" \
            --max_length 128 \
            --save_every 50 \
            --log_every  10 \
            --warmup_steps 30 \
            --anchor_weight 0.05 \
            $teacher_only_flag \
            $RESUME \
            >> "$out_dir/train.log" 2>&1

        EXIT=$?
        LATEST=$(ls "$out_dir"/simper_step*/pytorch_model.bin \
                    "$out_dir"/simper_step*/model.safetensors 2>/dev/null \
                 | sed 's|/[^/]*$||' | sort -t_ -k3 -n | tail -1 || true)
        STEP_DONE=$(echo "$LATEST" | grep -oP 'step\K[0-9]+' 2>/dev/null || echo 0)

        if [ "$EXIT" -eq 0 ] || [ "${STEP_DONE:-0}" -ge "$STEPS" ]; then
            echo "[S1] $name 训练完成 (step=$STEP_DONE)" | tee -a "$LOG"
            return 0
        fi
        echo "[S1] $name 失败 (attempt $attempt, exit=$EXIT)，30s 后重试" | tee -a "$LOG"
        sleep 30
    done
    echo "[S1] $name 连续失败，放弃" | tee -a "$LOG"
    return 1
}

# 实验1: v2_full（pre-filtered，all teacher_correct=True，teacher_only flag 无影响）
run_simper "simper_abl_v2_full" \
    "$DATA/abl_v2_full.jsonl" \
    "--teacher_only"

# 实验2: A0_11101k（等量无过滤控制组，--no-teacher_only 使用全量 construction direction）
run_simper "simper_abl_A0_11101k" \
    "$DATA/abl_A0_11101k_nofilter.jsonl" \
    "--no-teacher_only"

echo "" | tee -a "$LOG"
echo "[S1] 全部训练完成" | tee -a "$LOG"
echo "[S1] 下一步：对 simper_abl_v2_full/simper_step300 和 simper_abl_A0_11101k/simper_step300 运行全评测" | tee -a "$LOG"
