# Training

1. Copy [`configs/training/dataset_paths.example.env`](../configs/training/dataset_paths.example.env) to `dataset_paths.env` and fill in local paths.
2. Choose or adapt a launch profile from [`configs/training/launch_profiles/`](../configs/training/launch_profiles).
3. The hyperparameters are in [`configs/training/cpt.env`](../configs/training/cpt.env); the machine-readable manifest is at [`configs/training/cpt.regime.json`](../configs/training/cpt.regime.json).
4. For physical-order curriculum runs, apply/check the Megatron GPTDataset no-shuffle patch with [`scripts/train/patch_megatron_gpt_dataset_no_shuffle.py`](../scripts/train/patch_megatron_gpt_dataset_no_shuffle.py).
5. Launch [`scripts/train/train_apertus_cpt.sbatch`](../scripts/train/train_apertus_cpt.sbatch) directly with `sbatch`, or use [`scripts/train/submit_cpt_chain.sh`](../scripts/train/submit_cpt_chain.sh) for chained segments.

The trainer wraps Megatron through [`scripts/runtime/pretrain_gpt_te_guard.py`](../scripts/runtime/pretrain_gpt_te_guard.py); see [`MODEL_BRIDGE.md`](MODEL_BRIDGE.md) for what the guard handles.

## Run Scope

The full run is CPT with the tokenizer extension over the entire HPLT slice, the entire GlossAPI Greek slice, and replay data. Smaller runs, including 5B-token first-slice runs and TD-layer probes, are diagnostics.

[`configs/training/cpt.env`](../configs/training/cpt.env) holds the validated full-run optimizer, loss, batch, geometry, curriculum-order, and runtime values. Diagnostics should override run length or early-exit behavior explicitly, and record any intentional LR-schedule deviation.

## Regime vs Launch Profile

Treat tokenizer files, model geometry, optimizer, scheduler policies, loss, data semantics, and checkpoint semantics as the training regime. These are the choices to preserve unless the experiment is explicitly changing them.

Treat node count, walltime, partition/account, software image, and transport knobs as launch-profile choices. They should be adapted to the cluster, queue, budget, and target wall time. The Clariden 16-node profile records a validated setup; it is not a claim that 16 nodes are required.

## Required Inputs

`train_apertus_cpt.sbatch` requires:

- `ARM` — `vanilla`, `extension`, or `td`
- `INIT_CKPT` — Megatron-format initialization checkpoint
- `OUTPUT_DIR` — run output directory
- `MEGATRON_LM_DIR` — Megatron-LM-Swiss-AI checkout
- tokenizer and data-prefix paths from the sourced config files

The launcher writes `run_metadata.json` and `training_command.sh` into `OUTPUT_DIR`.

Building the per-arm `INIT_CKPT` itself — a vanilla embedding resize, or the Token-Distillation / centroid / ReTok new-token initialization for the `extension` and `td` arms — happens upstream and is out of scope for this repository. [`MODEL_BRIDGE.md`](MODEL_BRIDGE.md) covers the HF↔Megatron conversion the init checkpoint is built on.

## Curriculum Order

`CURRICULUM_ORDER_MODE=physical_order` is the default. It requires a small patch to Megatron's GPTDataset builder so the train `document_index` is sequential and the train `shuffle_index` is identity:

```bash
python3 scripts/train/patch_megatron_gpt_dataset_no_shuffle.py \
  --megatron-dir "$MEGATRON_LM_DIR"
python3 scripts/train/patch_megatron_gpt_dataset_no_shuffle.py --check \
  --megatron-dir "$MEGATRON_LM_DIR"
```

After a smoke run has built Megatron's dataset cache, verify the generated train indices:

```bash
python3 scripts/train/verify_megatron_curriculum_indices.py \
  --data-prefix "$EXT_DATA_PREFIX"
```

## 16-Node Clariden Profile

The reusable multi-node profile is encoded in [`configs/training/launch_profiles/clariden_16node_cxi.env`](../configs/training/launch_profiles/clariden_16node_cxi.env), [`scripts/train/submit_cpt_chain.sh`](../scripts/train/submit_cpt_chain.sh), and [`scripts/train/train_apertus_cpt.sbatch`](../scripts/train/train_apertus_cpt.sbatch).

The validated setup, in brief:

- The validated example used `NODES=16`, `GPUS_PER_NODE=4`, tensor parallel 2, pipeline parallel 1.
- The profile uses checkpoint-boundary segments: `SEGMENTS=4`, `EXIT_INTERVAL=952`.
- Multi-node launches use `LAUNCH_MODE=torchrun`.
- Slurm starts one task per node: `--ntasks-per-node=1`; `torchrun` starts the four local GPU workers.
- The trainer sets `WORLD_SIZE = nodes x GPUS_PER_NODE` in torchrun mode, not `SLURM_NTASKS`.
- The transport profile uses `NCCL_NET="AWS Libfabric"` with CXI/libfabric defaults and `NCCL_NET_FORCE_FLUSH=0`.
- Runtime transport variables are re-exported inside the `uenv run` shell so uenv defaults cannot silently replace them.

Tune node count, walltime, segment count, and exit interval for the current run. The older one-task-per-GPU Slurm shape is still supported for one-node runs, but it is not the default multi-node profile.

## Preflight Checks

`train_apertus_cpt.sbatch` fails early if:

- `TRAIN_TOKENS < LR_WARMUP_TOKENS`
- Slurm launch mode did not start `nodes × GPUS_PER_NODE` tasks
- the required tokenizer or data-prefix variables are missing
- `CURRICULUM_ORDER_MODE=physical_order` is selected but the Megatron no-shuffle patch is absent

For short diagnostic runs, keep `TRAIN_TOKENS` large enough to cover warmup and stop early with `EXIT_INTERVAL`.

## Hyperparameters

See [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md).
