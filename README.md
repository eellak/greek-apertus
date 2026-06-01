# Greek Apertus

Continued pretraining of [Apertus-8B](https://huggingface.co/swiss-ai/Apertus-8B-2509) on Greek, with the tokenizer extension and the model-format bridge that go with it.

Contents:

- **Hyperparameters** — env file and JSON manifest for the CPT recipe (optimizer, loss, positional geometry, data semantics, parallelism, runtime guards). See [`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md).
- **Training harness** — Slurm/Megatron entrypoints with hard preflight checks. See [`docs/TRAINING.md`](docs/TRAINING.md).
- **HF ↔ Megatron bridge** — round-trip conversion plus xIELU and QK-Norm fidelity patching. See [`docs/MODEL_BRIDGE.md`](docs/MODEL_BRIDGE.md).
- **Tokenizer tooling** — append-only Greek cutoff variants and added-token removal manifests. See [`docs/TOKENIZER_EXTENSION.md`](docs/TOKENIZER_EXTENSION.md).

## Layout

- [`configs/training/`](configs/training) — `cpt.env`, `cpt.regime.json`, and a `dataset_paths.example.env` scaffold.
- [`configs/tokenizer/`](configs/tokenizer) — cutoff grid and removal policy.
- [`scripts/train/`](scripts/train) — training entrypoints.
- [`scripts/model_bridge/`](scripts/model_bridge) — HF ↔ Megatron-LM-Swiss-AI conversion tools.
- [`scripts/runtime/`](scripts/runtime) — training-launcher runtime guard.
- [`scripts/tokenizer/`](scripts/tokenizer) — cutoff builder and removal-manifest emitter.

## License

License pending. Until a `LICENSE` file is added, the repository is not licensed for reuse.
