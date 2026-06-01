#!/usr/bin/env bash
#SBATCH --job-name=apertus_cpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR2@600
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="$(cd "${REPO_ROOT_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"
TRAIN_CONFIG="${TRAIN_CONFIG:-$REPO_ROOT/configs/training/cpt.env}"
source "$TRAIN_CONFIG"
if [ -n "${DATASET_PATHS_ENV:-}" ]; then
    source "$DATASET_PATHS_ENV"
fi

ARM="${ARM:?ARM is required: vanilla, extension, or td}"
INIT_CKPT="${INIT_CKPT:?INIT_CKPT is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:?MEGATRON_LM_DIR is required}"

case "$ARM" in
    vanilla)
        ARM_TOKENIZER="${BASE_TOKENIZER_DIR:?BASE_TOKENIZER_DIR is required for vanilla}"
        ARM_DATA_PREFIX="${BASE_DATA_PREFIX:?BASE_DATA_PREFIX is required for vanilla}"
        ;;
    extension|td)
        ARM_TOKENIZER="${EXT_TOKENIZER_DIR:?EXT_TOKENIZER_DIR is required for extension/td}"
        ARM_DATA_PREFIX="${EXT_DATA_PREFIX:?EXT_DATA_PREFIX is required for extension/td}"
        ;;
    *)
        echo "ERROR: ARM=$ARM not recognized (expected vanilla|extension|td)" >&2
        exit 2
        ;;
esac

SEQ_LEN="$SEQ_LENGTH"
TRAIN_SAMPLES=$(( TRAIN_TOKENS / SEQ_LEN ))
LR_WARMUP_SAMPLES=$(( LR_WARMUP_TOKENS / SEQ_LEN ))
MODEL_WORLD_SIZE=$(( ${SLURM_NNODES:-$NODES} * GPUS_PER_NODE ))

if (( TRAIN_TOKENS < LR_WARMUP_TOKENS )); then
    echo "ERROR: TRAIN_TOKENS=$TRAIN_TOKENS is smaller than LR_WARMUP_TOKENS=$LR_WARMUP_TOKENS" >&2
    echo "For probes, keep TRAIN_TOKENS at the notional full target and set EXIT_INTERVAL." >&2
    exit 2
fi

if [ "${LAUNCH_MODE:-slurm}" = "slurm" ] && [ -n "${SLURM_NTASKS:-}" ] && [ "$SLURM_NTASKS" -ne "$MODEL_WORLD_SIZE" ]; then
    echo "ERROR: Slurm launched SLURM_NTASKS=$SLURM_NTASKS but expected $MODEL_WORLD_SIZE." >&2
    echo "Submit with --ntasks-per-node=$GPUS_PER_NODE so WORLD_SIZE matches the GPU count." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/tensorboard" "$OUTPUT_DIR/triggers"
TRIGGER_DIR="$OUTPUT_DIR/triggers"
rm -f "$TRIGGER_DIR/exit" "$TRIGGER_DIR/save"

echo "=== train_apertus_cpt.sbatch ==="
date -u
echo "ARM=$ARM"
echo "INIT_CKPT=$INIT_CKPT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TRAIN_CONFIG=$TRAIN_CONFIG"
echo "MEGATRON_LM_DIR=$MEGATRON_LM_DIR @ ${MEGATRON_LM_COMMIT:-unknown}"
echo "tokenizer=$ARM_TOKENIZER"
echo "data_prefix=$ARM_DATA_PREFIX"
echo "train_tokens=$TRAIN_TOKENS train_samples=$TRAIN_SAMPLES warmup_samples=$LR_WARMUP_SAMPLES"
echo "geometry: max_position=$MAX_POSITION_EMBEDDINGS rotary_base=$ROTARY_BASE rope_scaling=$USE_ROPE_SCALING factor=$ROPE_SCALING_FACTOR"
echo

TRANSFORMER_ENGINE_ARGS=(
    --main-grads-dtype "$MAIN_GRADS_DTYPE"
)

NETWORK_SIZE_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 21504
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --position-embedding-type rope
    --rotary-base "$ROTARY_BASE"
    --make-vocab-size-divisible-by 128
    --normalization "$NORMALIZATION"
    --xielu
    --qk-layernorm
    --qknorm-impl "$QKNORM_IMPL"
    --untie-embeddings-and-output-weights
)
if [ "$USE_ROPE_SCALING" = "1" ]; then
    NETWORK_SIZE_ARGS+=(--use-rope-scaling --rope-scaling-factor "$ROPE_SCALING_FACTOR")
fi

LOGGING_ARGS=(
    --log-throughput
    --log-params-norm
    --tensorboard-dir "$OUTPUT_DIR/tensorboard"
    --no-log-loss-scale-to-tensorboard
    --log-memory-to-tensorboard
    --log-interval 1
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay "$ADEMA_WEIGHT_DECAY"
    --clip-grad "$GRAD_CLIP"
    --adam-beta1 "$ADEMA_BETA1"
    --adam-beta2 "$ADEMA_BETA2"
    --ademamix-alpha "$ADEMA_ALPHA"
    --ademamix-beta3 "$ADEMA_BETA3"
    --ademamix-beta3-warmup "$ADEMA_BETA3_WARMUP_STEPS"
    --ademamix-alpha-warmup "$ADEMA_ALPHA_WARMUP_STEPS"
)

TRAINING_ARGS=(
    --micro-batch-size "$MICROBATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --no-check-for-nan-in-loss-and-grad
    --train-samples "$TRAIN_SAMPLES"
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer ademamix
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 500
    --exit-signal-handler
    --trigger-path "$TRIGGER_DIR"
)
if [ -n "$EXIT_INTERVAL" ]; then
    TRAINING_ARGS+=(--exit-interval "$EXIT_INTERVAL")
fi
if [ "$LOSS_OBJECTIVE" = "goldfish" ]; then
    TRAINING_ARGS+=(--goldfish-loss --goldfish-k "$GOLDFISH_K" --goldfish-h "$GOLDFISH_H")
elif [ "$LOSS_OBJECTIVE" != "ntp" ]; then
    echo "ERROR: LOSS_OBJECTIVE=$LOSS_OBJECTIVE not recognized (expected goldfish|ntp)" >&2
    exit 2
fi

INITIALIZATION_ARGS=(
    --seed "$DATA_SEED"
    --init-method-std "$INIT_STD"
)

LEARNING_RATE_ARGS=(
    --lr "$LR_PEAK"
    --min-lr "$LR_FINAL"
    --lr-decay-style "$LR_SCHEDULE_STYLE"
    --lr-warmup-samples "$LR_WARMUP_SAMPLES"
    --lr-warmup-init "$LR_WARMUP_INIT"
)
if [ "$LR_SCHEDULE_STYLE" = "WSD" ]; then
    LEARNING_RATE_ARGS+=(--lr-wsd-decay-style "$LR_DECAY_STYLE" --lr-wsd-decay-samples "$TRAIN_SAMPLES")
fi

CHECKPOINTING_ARGS=(
    --load "$INIT_CKPT"
    --ckpt-format torch_dist
    --ckpt-fully-parallel-load
    --dist-ckpt-strictness assume_ok_unexpected
    --override-opt_param-scheduler
    --distributed-timeout-minutes 60
)
if [ "$DISABLE_SAVE" = "0" ]; then
    CHECKPOINTING_ARGS+=(--save "$OUTPUT_DIR/checkpoints" --save-interval "$SAVE_INTERVAL" --async-save)
elif [ "$DISABLE_SAVE" != "1" ]; then
    echo "ERROR: DISABLE_SAVE=$DISABLE_SAVE not recognized (expected 0|1)" >&2
    exit 2
fi
if [ "$RESUME_TRAINING" = "0" ]; then
    CHECKPOINTING_ARGS+=(--no-load-optim --no-load-rng)
elif [ "$RESUME_TRAINING" != "1" ]; then
    echo "ERROR: RESUME_TRAINING=$RESUME_TRAINING not recognized (expected 0|1)" >&2
    exit 2
fi

MIXED_PRECISION_ARGS=("$MIXED_PRECISION_FLAG")
DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size "$TENSOR_MODEL_PARALLEL_SIZE"
    --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE"
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TOKENIZER_ARGS=(--tokenizer-type HuggingFaceTokenizer --tokenizer-model "$ARM_TOKENIZER")
DATA_ARGS=(
    --split 100,0,0
    --seq-length "$SEQ_LEN"
    --num-workers "$DATALOADER_WORKERS"
    --num-dataset-builder-threads 4
    --data-path "$ARM_DATA_PREFIX"
)
[ "$RESET_ATTENTION_MASK" = "1" ] && DATA_ARGS+=(--reset-attention-mask)
[ "$RESET_POSITION_IDS" = "1" ] && DATA_ARGS+=(--reset-position-ids)
[ "$EOD_MASK_LOSS" = "1" ] && DATA_ARGS+=(--eod-mask-loss)

RUNTIME_GUARD="$REPO_ROOT/scripts/runtime/pretrain_gpt_te_guard.py"
TRAINING_CMD=(
    python3 "$RUNTIME_GUARD"
    "$MEGATRON_LM_DIR/pretrain_gpt.py"
    "${TRANSFORMER_ENGINE_ARGS[@]}"
    "${NETWORK_SIZE_ARGS[@]}"
    "${LOGGING_ARGS[@]}"
    "${REGULARIZATION_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${INITIALIZATION_ARGS[@]}"
    "${LEARNING_RATE_ARGS[@]}"
    "${CHECKPOINTING_ARGS[@]}"
    "${MIXED_PRECISION_ARGS[@]}"
    "${DISTRIBUTED_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    "${DATA_ARGS[@]}"
)
printf -v TRAINING_CMD_Q '%q ' "${TRAINING_CMD[@]}"
printf '%s\n' "$TRAINING_CMD_Q" > "$OUTPUT_DIR/training_command.sh"

cat > "$OUTPUT_DIR/run_metadata.json" <<META
{
  "arm": "$ARM",
  "init_ckpt": "$INIT_CKPT",
  "output_dir": "$OUTPUT_DIR",
  "megatron_dir": "$MEGATRON_LM_DIR",
  "megatron_commit": "${MEGATRON_LM_COMMIT:-unknown}",
  "seq_length": $SEQ_LEN,
  "global_batch_samples": $GLOBAL_BATCH_SIZE,
  "global_batch_tokens": $GLOBAL_BATCH_TOKENS,
  "train_tokens": $TRAIN_TOKENS,
  "lr_peak": "$LR_PEAK",
  "lr_final": "$LR_FINAL",
  "lr_schedule_style": "$LR_SCHEDULE_STYLE",
  "lr_warmup_tokens": $LR_WARMUP_TOKENS,
  "loss_objective": "$LOSS_OBJECTIVE",
  "max_position_embeddings": $MAX_POSITION_EMBEDDINGS,
  "rotary_base": $ROTARY_BASE,
  "use_rope_scaling": $USE_ROPE_SCALING,
  "rope_scaling_factor": "$ROPE_SCALING_FACTOR",
  "tensor_model_parallel_size": $TENSOR_MODEL_PARALLEL_SIZE,
  "pipeline_model_parallel_size": $PIPELINE_MODEL_PARALLEL_SIZE,
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
META

export PYTHONPATH="$MEGATRON_LM_DIR"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$PYTORCH_ALLOC_CONF}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export FI_CXI_DEFAULT_TX_SIZE=16384
export NCCL_NET_FORCE_FLUSH=1
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
export NCCL_RAS_ENABLE=0
export CUDA_CACHE_DISABLE=1

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((32000 + (${SLURM_JOB_ID:-0} % 1000)))}"
export WORLD_SIZE="${SLURM_NTASKS:-$MODEL_WORLD_SIZE}"
echo "distributed: MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

LAUNCH_PREFIX=()
if [ -n "${UENV_IMAGE:-}" ]; then
    if ! command -v uenv >/dev/null 2>&1; then
        echo "ERROR: UENV_IMAGE is set but uenv is not on PATH" >&2
        exit 2
    fi
    LAUNCH_PREFIX=(uenv run "$UENV_IMAGE" --view="$UENV_VIEW" --)
fi

cd "$MEGATRON_LM_DIR"
case "${LAUNCH_MODE:-slurm}" in
    slurm)
        "${LAUNCH_PREFIX[@]}" srun --cpus-per-task "${SLURM_CPUS_PER_TASK:-4}" \
            -lu bash -c "export PYTHONPATH='$MEGATRON_LM_DIR' RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID; exec $TRAINING_CMD_Q"
        ;;
    torchrun)
        "${LAUNCH_PREFIX[@]}" srun --ntasks-per-node=1 --cpus-per-task "${SLURM_CPUS_PER_TASK:-4}" \
            -lu bash -c "export PYTHONPATH='$MEGATRON_LM_DIR' MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT'; exec torchrun --no-python --nnodes='${SLURM_NNODES:-$NODES}' --nproc_per_node='$GPUS_PER_NODE' --node_rank=\$SLURM_PROCID --rdzv_backend=c10d --rdzv_endpoint='$MASTER_ADDR:$MASTER_PORT' $TRAINING_CMD_Q"
        ;;
    *)
        echo "ERROR: LAUNCH_MODE=$LAUNCH_MODE not recognized (expected slurm|torchrun)" >&2
        exit 2
        ;;
esac
