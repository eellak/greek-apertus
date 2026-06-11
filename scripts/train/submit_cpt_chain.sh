#!/usr/bin/env bash
set -euo pipefail

# Generic chained submitter for Apertus CPT runs. DRY_RUN=1 by default so the
# generated sbatch commands can be inspected before touching a cluster queue.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/scripts/train/train_apertus_cpt.sbatch}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$REPO_ROOT/configs/training/cpt.env}"
LAUNCH_PROFILE_ENV="${LAUNCH_PROFILE_ENV:-}"
DATASET_PATHS_ENV="${DATASET_PATHS_ENV:-$REPO_ROOT/configs/training/dataset_paths.env}"

if [ -n "$LAUNCH_PROFILE_ENV" ]; then
    case "$LAUNCH_PROFILE_ENV" in
        /*) ;;
        *) LAUNCH_PROFILE_ENV="$REPO_ROOT/$LAUNCH_PROFILE_ENV" ;;
    esac
    source "$LAUNCH_PROFILE_ENV"
fi
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
CONFIRM_LAUNCH="${CONFIRM_LAUNCH:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_TAG}"
STATE_DIR="${STATE_DIR:-$OUTPUT_DIR/submit_state}"

case "$DRY_RUN" in 0|1) ;; *) echo "ERROR: DRY_RUN must be 0|1" >&2; exit 2 ;; esac
if [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_LAUNCH" != "1" ]; then
    echo "ERROR: live launch requires CONFIRM_LAUNCH=1" >&2
    exit 2
fi

if (( TRAIN_TOKENS < LR_WARMUP_TOKENS )); then
    echo "ERROR: TRAIN_TOKENS=$TRAIN_TOKENS is smaller than LR_WARMUP_TOKENS=$LR_WARMUP_TOKENS" >&2
    echo "For probes, keep TRAIN_TOKENS large enough to cover warmup and set EXIT_INTERVAL." >&2
    exit 2
fi

if [ "$ARM" = "vanilla" ]; then
    : "${BASE_DATA_PREFIX:?BASE_DATA_PREFIX is required}"
    : "${BASE_TOKENIZER_DIR:?BASE_TOKENIZER_DIR is required}"
else
    : "${EXT_DATA_PREFIX:?EXT_DATA_PREFIX is required}"
    : "${EXT_TOKENIZER_DIR:?EXT_TOKENIZER_DIR is required}"
fi

if [ "$SEGMENTS" -gt 1 ] && [ -z "${EXIT_INTERVAL:-}" ]; then
    echo "ERROR: SEGMENTS=$SEGMENTS requires EXIT_INTERVAL so each segment stops at a deterministic checkpoint boundary." >&2
    exit 2
fi
if [ -n "${EXIT_INTERVAL:-}" ] && [ $(( EXIT_INTERVAL % SAVE_INTERVAL )) -ne 0 ]; then
    echo "ERROR: EXIT_INTERVAL=$EXIT_INTERVAL must be a multiple of SAVE_INTERVAL=$SAVE_INTERVAL." >&2
    exit 2
fi

SBATCH_NTASKS_PER_NODE="$GPUS_PER_NODE"
if [ "${LAUNCH_MODE:-slurm}" = "torchrun" ]; then
    SBATCH_NTASKS_PER_NODE="1"
fi

mkdir -p "$STATE_DIR"
CHAIN_TSV="$STATE_DIR/chain.tsv"
printf "arm\tsegment\tresume\tinit_ckpt\tdependency\tjob_id\n" > "$CHAIN_TSV"

echo "=== submit_cpt_chain.sh ==="
echo "ARM:                  $ARM"
echo "TRAIN_CONFIG:         $TRAIN_CONFIG"
echo "LAUNCH_PROFILE_ENV:   ${LAUNCH_PROFILE_ENV:-<none>}"
echo "OUTPUT_DIR:           $OUTPUT_DIR"
echo "TRAIN_TOKENS:         $TRAIN_TOKENS"
echo "SCHEDULER:            $LR_SCHEDULE_STYLE warmup_iters=$LR_WARMUP_ITERS wsd_decay_samples=${LR_WSD_DECAY_SAMPLES:-n/a}"
echo "CURRICULUM_ORDER:     ${CURRICULUM_ORDER_MODE:-randomized}"
echo "SEGMENTS:             $SEGMENTS"
echo "LAUNCH_MODE:          ${LAUNCH_MODE:-slurm} sbatch_ntasks_per_node=$SBATCH_NTASKS_PER_NODE"
echo "TRANSPORT_PROFILE:    ${TRANSPORT_PROFILE:-generic}"
echo "DRY_RUN:              $DRY_RUN"
echo

prev_job=""
for segment in $(seq 1 "$SEGMENTS"); do
    if [ "$segment" = "1" ]; then
        resume=0
        init_for_segment="$INIT_CKPT"
        dep_args=()
        dep_label="none"
    else
        resume=1
        init_for_segment="$OUTPUT_DIR/checkpoints"
        dep_args=(--dependency "afterok:$prev_job")
        dep_label="$prev_job"
    fi

    [ "$DRY_RUN" = "0" ] && mkdir -p "$OUTPUT_DIR"

    sbatch_args=(
        --nodes "$NODES"
        --ntasks-per-node "$SBATCH_NTASKS_PER_NODE"
        --gpus-per-node "$GPUS_PER_NODE"
        --gres "gpu:$GPUS_PER_NODE"
        --cpus-per-task "${CPUS_PER_TASK:-36}"
        --time "$TIME_LIMIT"
        --job-name "apertus_${ARM}_${RUN_TAG}_${segment}"
        --output "$OUTPUT_DIR/slurm-%x-%j.out"
        --export "ALL,REPO_ROOT_OVERRIDE=$REPO_ROOT,TRAIN_CONFIG=$TRAIN_CONFIG,LAUNCH_PROFILE_ENV=$LAUNCH_PROFILE_ENV,DATASET_PATHS_ENV=$DATASET_PATHS_ENV,ARM=$ARM,INIT_CKPT=$init_for_segment,OUTPUT_DIR=$OUTPUT_DIR,TRAIN_TOKENS=$TRAIN_TOKENS,RESUME_TRAINING=$resume,SAVE_INTERVAL=$SAVE_INTERVAL,EXIT_INTERVAL=${EXIT_INTERVAL:-},LAUNCH_MODE=${LAUNCH_MODE:-slurm},NODES=$NODES,GPUS_PER_NODE=$GPUS_PER_NODE"
    )
    if [ -n "${PARTITION:-}" ]; then
        sbatch_args+=(--partition "$PARTITION")
    fi
    if [ -n "${ACCOUNT:-}" ]; then
        sbatch_args+=(--account "$ACCOUNT")
    fi
    sbatch_args+=("${dep_args[@]}")

    if [ "$DRY_RUN" = "1" ]; then
        printf 'sbatch'
        printf ' %q' "${sbatch_args[@]}" "$TRAIN_SCRIPT"
        printf '\n'
        job_id="DRYRUN_SEGMENT_$segment"
    else
        job_id="$(sbatch --parsable "${sbatch_args[@]}" "$TRAIN_SCRIPT")"
        echo "submitted segment $segment: $job_id resume=$resume dep=$dep_label"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$ARM" "$segment" "$resume" "$init_for_segment" "$dep_label" "$job_id" >> "$CHAIN_TSV"
    prev_job="$job_id"
done

echo
echo "chain manifest: $CHAIN_TSV"
