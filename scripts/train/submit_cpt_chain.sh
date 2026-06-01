#!/usr/bin/env bash
set -euo pipefail

# Generic chained submitter for Apertus CPT runs. DRY_RUN=1 by default so the
# generated sbatch commands can be inspected before touching a cluster queue.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/scripts/train/train_apertus_cpt.sbatch}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$REPO_ROOT/configs/training/cpt.env}"
DATASET_PATHS_ENV="${DATASET_PATHS_ENV:-$REPO_ROOT/configs/training/dataset_paths.env}"

source "$TRAIN_CONFIG"
if [ -f "$DATASET_PATHS_ENV" ]; then
    source "$DATASET_PATHS_ENV"
fi

ARM="${ARM:?ARM is required: vanilla, extension, or td}"
INIT_CKPT="${INIT_CKPT:?INIT_CKPT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
SEGMENTS="${SEGMENTS:-1}"
DRY_RUN="${DRY_RUN:-1}"

if (( TRAIN_TOKENS < LR_WARMUP_TOKENS )); then
    echo "ERROR: TRAIN_TOKENS=$TRAIN_TOKENS is smaller than LR_WARMUP_TOKENS=$LR_WARMUP_TOKENS" >&2
    echo "For probes, keep TRAIN_TOKENS at the notional full target and set EXIT_INTERVAL." >&2
    exit 2
fi

if [ "$ARM" = "vanilla" ]; then
    : "${BASE_DATA_PREFIX:?BASE_DATA_PREFIX is required}"
    : "${BASE_TOKENIZER_DIR:?BASE_TOKENIZER_DIR is required}"
else
    : "${EXT_DATA_PREFIX:?EXT_DATA_PREFIX is required}"
    : "${EXT_TOKENIZER_DIR:?EXT_TOKENIZER_DIR is required}"
fi

prev_job=""
for segment in $(seq 1 "$SEGMENTS"); do
    output_dir="$RUN_ROOT/$RUN_TAG/segment_$(printf '%02d' "$segment")"
    mkdir -p "$output_dir"

    sbatch_args=(
        --nodes "$NODES"
        --ntasks-per-node "$GPUS_PER_NODE"
        --gpus-per-node "$GPUS_PER_NODE"
        --cpus-per-task "${CPUS_PER_TASK:-36}"
        --time "$TIME_LIMIT"
        --partition "$PARTITION"
        --job-name "apertus_${ARM}_${RUN_TAG}_${segment}"
        --output "$output_dir/slurm-%x-%j.out"
        --export "ALL,REPO_ROOT_OVERRIDE=$REPO_ROOT,TRAIN_CONFIG=$TRAIN_CONFIG,DATASET_PATHS_ENV=$DATASET_PATHS_ENV,ARM=$ARM,INIT_CKPT=$INIT_CKPT,OUTPUT_DIR=$output_dir,TRAIN_TOKENS=$TRAIN_TOKENS,RESUME_TRAINING=$([ "$segment" = "1" ] && echo 0 || echo 1)"
    )
    if [ -n "${ACCOUNT:-}" ]; then
        sbatch_args+=(--account "$ACCOUNT")
    fi
    if [ -n "$prev_job" ]; then
        sbatch_args+=(--dependency "afterok:$prev_job")
    fi

    if [ "$DRY_RUN" = "1" ]; then
        printf 'sbatch'
        printf ' %q' "${sbatch_args[@]}" "$TRAIN_SCRIPT"
        printf '\n'
        prev_job="DRYRUN_SEGMENT_$segment"
    else
        job_id="$(sbatch --parsable "${sbatch_args[@]}" "$TRAIN_SCRIPT")"
        echo "submitted segment $segment: $job_id"
        prev_job="$job_id"
    fi
done
