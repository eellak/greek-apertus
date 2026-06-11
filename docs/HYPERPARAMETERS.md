# Hyperparameters

Hyperparameters for Greek Apertus CPT. "Hyperparameters" here spans optimizer, loss, positional geometry, data semantics, model parallelism, checkpoint behavior, and runtime guards — not just scalar LR/batch settings.

Executable config: [`configs/training/cpt.env`](../configs/training/cpt.env)

Machine-readable manifest: [`configs/training/cpt.regime.json`](../configs/training/cpt.regime.json)

## Scope

The full run is CPT with the tokenizer extension over the entire HPLT slice, the entire GlossAPI Greek slice, and replay data. Smaller runs are diagnostics and must state any overrides explicitly. The 5B-token runs exercised only the opening of training; their constant-after-warmup LR schedule is not the full-run regime.

## Training

| Field | Value |
| --- | ---: |
| Sequence length | 4096 |
| Global batch | 1024 samples (4.19 M tokens) |
| Microbatch | 2 |
| Optimizer | AdEMAMix |
| Adam β1 | 0.9 |
| Adam β2 | 0.995 † |
| AdEMAMix β3 | 0.999 |
| AdEMAMix α | 4.0 † |
| Weight decay | 0.1 |
| Gradient clipping | 0.1 |
| Peak LR | 5.5e-5 † |
| Final LR | 5.5e-6 |
| LR schedule | WSD, 1-sqrt cooldown |
| Warmup init LR | 5.5e-6 |
| Warmup | 400 iterations, ~1.68 B tokens |
| Cooldown | final 20% of training samples |
| Train tokens | 13.5 B |
| Loss | Goldfish (k = 50, h = 50) |
| Tensor parallel | 2 |
| Pipeline parallel | 1 |
| Precision | bf16, fp32 main grads |

† Starting value, not a frozen result — subject of a planned sweep, not yet run. β2 = 0.995 (sweep range [0.99, 0.999], couples to warmup); α = 4.0 (≈ half of Apertus-8B's 8); peak LR 5.5e-5 = 0.5 × Apertus's pretraining peak (1.1e-4). Do not replicate these three as settled. The machine manifest records the same under `open_decisions.sweep_candidates`.

## Non-Scalar Choices

- Loss: Goldfish, `k = 50`, `h = 50`.
- Optimizer: AdEMAMix with `β2 = 0.995`, `β3 = 0.999`, `α = 4.0`, β3 half-life warmup, and α linear warmup over the full run.
- Architecture: xIELU, QK-LayerNorm, RMSNorm, untied embeddings/output weights, bias-free linear layers.
- Data semantics: reset attention mask, reset position ids, EOD mask loss, single-dataset split `100,0,0`.
- Curriculum order: `CURRICULUM_ORDER_MODE=physical_order` requires the Megatron GPTDataset no-shuffle patch; use `randomized` only to reproduce randomized-sampler runs.
- Runtime: Transformer Engine guard, torch-dist metadata fallback, xIELU optimizer audit.
- Model parallel shape: tensor parallel 2, pipeline parallel 1.
- Resource launch shape: use a launch profile. Node count, walltime, account, partition, software image, and transport knobs are operational choices, not hyperparameters.

## Scheduler Policies

Schedulers are stored as policies in [`configs/training/cpt.regime.json`](../configs/training/cpt.regime.json), with resolved values for the default 13.5B-token run.

| Scheduler | Policy |
| --- | --- |
| LR warmup | `round(2 / (1 - β2))` iterations; 400 iterations for β2 = 0.995 |
| LR shape | WSD, same shape independent of corpus mixture |
| LR cooldown | final 20% of training samples, `1-sqrt` shape |
| LR floor | `0.1 x peak_lr` |
| β3 warmup | Megatron half-life warmup from β1 to β3 over `TRAIN_ITERS` |
| α warmup | linear warmup over `TRAIN_ITERS` |

The formulas are the frozen regime; the resolved token/sample counts are derived and shift when `TRAIN_TOKENS` or batch size changes.

## Positional Geometry

| Field | Value |
| --- | ---: |
| Max position embeddings | 4096 |
| Rotary base | 500 000 |
| RoPE scaling | enabled |
| RoPE scaling factor | 8.0 |

## Probe Pattern

For short diagnostic runs, keep `TRAIN_TOKENS` at or above `LR_WARMUP_TOKENS` and use `EXIT_INTERVAL` to stop early. Warmup is samples-based, so the scheduler stays valid. If a diagnostic intentionally uses a different LR schedule, record that as a diagnostic override.
