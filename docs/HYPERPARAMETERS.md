# Hyperparameters

The hyperparameters for Greek Apertus CPT. "Hyperparameters" here covers the full training contract: optimizer, loss, positional geometry, data semantics, model parallelism, checkpoint behavior, and runtime guards — not only scalar LR/batch settings.

Executable config: [`configs/training/cpt.env`](../configs/training/cpt.env)

Machine-readable manifest: [`configs/training/cpt.regime.json`](../configs/training/cpt.regime.json)

## Training

| Field | Value |
| --- | ---: |
| Sequence length | 4096 |
| Global batch | 1024 samples (4.19 M tokens) |
| Microbatch | 2 |
| Optimizer | AdEMAMix |
| Adam β1 | 0.9 |
| Adam β2 | 0.999 |
| AdEMAMix β3 | 0.99 |
| AdEMAMix α | 8.0 |
| Weight decay | 0.1 |
| Gradient clipping | 0.1 |
| Peak LR | 1.1e-5 |
| Final LR | 1.1e-5 |
| LR schedule | constant after warmup |
| Warmup init LR | 1.1e-6 |
| Warmup tokens | 1.2 B |
| Loss | Goldfish (k = 50, h = 50) |
| Tensor parallel | 2 |
| Pipeline parallel | 1 |
| Precision | bf16, fp32 main grads |

## Non-Scalar Choices

- Loss: Goldfish, `k = 50`, `h = 50`.
- Optimizer: AdEMAMix with `β3 = 0.99` and `α = 8.0`.
- Architecture: xIELU, QK-LayerNorm, RMSNorm, untied embeddings/output weights, bias-free linear layers.
- Data semantics: reset attention mask, reset position ids, EOD mask loss, single-dataset split `100,0,0`.
- Runtime: Transformer Engine guard, torch-dist metadata fallback, xIELU optimizer audit.
- Distributed shape: tensor parallel 2, pipeline parallel 1, Slurm `--ntasks-per-node` equal to `GPUS_PER_NODE`.

## Positional Geometry

| Field | Value |
| --- | ---: |
| Max position embeddings | 65 536 |
| Rotary base | 12 000 000 |
| RoPE scaling | enabled |
| RoPE scaling factor | 8.0 |

## Probe Pattern

For short diagnostic runs, keep notional `TRAIN_TOKENS` at or above `LR_WARMUP_TOKENS` and use `EXIT_INTERVAL` to stop early. Warmup is samples-based, so the scheduler stays valid.
