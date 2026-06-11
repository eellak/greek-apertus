# Greek Apertus

Continued pretraining of [Apertus-8B](https://huggingface.co/swiss-ai/Apertus-8B-2509) on Greek: the tokenizer extension, the HF ↔ Megatron model bridge, and the training harness that drives the run.

Contents:

- **Hyperparameters** — env file and JSON manifest for the CPT regime (optimizer, loss, positional geometry, data semantics, parallelism, runtime guards). See [`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md).
- **Training harness** — Slurm/Megatron entrypoints, scheduler policy, launch profiles, and curriculum-order guards. See [`docs/TRAINING.md`](docs/TRAINING.md).
- **HF ↔ Megatron bridge** — round-trip conversion plus xIELU and QK-Norm fidelity patching. See [`docs/MODEL_BRIDGE.md`](docs/MODEL_BRIDGE.md).
- **Tokenizer tooling** — the cleaned 17,408-unit Greek extension builder and added-token removal manifest. See [`docs/TOKENIZER_EXTENSION.md`](docs/TOKENIZER_EXTENSION.md).
- **Reuse boundary** — which settings are frozen design decisions and which are per-run knobs to retune. See [`docs/REUSE_GUIDE.md`](docs/REUSE_GUIDE.md).

## Layout

- [`configs/training/`](configs/training) — training regime, launch profiles, and a `dataset_paths.example.env` template.
- [`configs/tokenizer/`](configs/tokenizer) — final extension contract and removal policy.
- [`scripts/train/`](scripts/train) — training entrypoints.
- [`scripts/model_bridge/`](scripts/model_bridge) — HF ↔ Megatron-LM-Swiss-AI conversion tools.
- [`scripts/runtime/`](scripts/runtime) — training-launcher runtime guard.
- [`scripts/tokenizer/`](scripts/tokenizer) — clean extension builder and removal-manifest emitter.

## License

[MIT](LICENSE) — © 2026 GFOSS – Open Technologies Alliance. Free to use, modify, and redistribute, including commercially; retain the copyright notice.
