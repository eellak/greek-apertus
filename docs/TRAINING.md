# Training

1. Copy [`configs/training/dataset_paths.example.env`](../configs/training/dataset_paths.example.env) to `dataset_paths.env` and fill in local paths.
2. The hyperparameters are in [`configs/training/cpt.env`](../configs/training/cpt.env); the machine-readable manifest is at [`configs/training/cpt.regime.json`](../configs/training/cpt.regime.json).
3. Launch [`scripts/train/train_apertus_cpt.sbatch`](../scripts/train/train_apertus_cpt.sbatch) directly with `sbatch`, or use [`scripts/train/submit_cpt_chain.sh`](../scripts/train/submit_cpt_chain.sh) for chained segments.

The trainer wraps Megatron through [`scripts/runtime/pretrain_gpt_te_guard.py`](../scripts/runtime/pretrain_gpt_te_guard.py); see [`MODEL_BRIDGE.md`](MODEL_BRIDGE.md) for what the guard handles.

## Required Inputs

`train_apertus_cpt.sbatch` requires:

- `ARM` — `vanilla`, `extension`, or `td`
- `INIT_CKPT` — Megatron-format initialization checkpoint
- `OUTPUT_DIR` — run output directory
- tokenizer and data-prefix paths from the sourced config files

The launcher writes `run_metadata.json` and `training_command.sh` into `OUTPUT_DIR`.

## Preflight Checks

The script fails early if:

- `TRAIN_TOKENS < LR_WARMUP_TOKENS`
- Slurm launched fewer tasks than `nodes × GPUS_PER_NODE`
- the required tokenizer or data-prefix variables are missing

For short diagnostic runs, keep `TRAIN_TOKENS` at the notional full target and stop early with `EXIT_INTERVAL`.

## Hyperparameters

See [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md).
