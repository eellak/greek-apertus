# Full Apertus-8B CPT settings

The authoritative machine recipe is
[`configs/training/full_8b_mixed_cpt.json`](../configs/training/full_8b_mixed_cpt.json).
[`cpt.env`](../configs/training/cpt.env) mirrors its scalar defaults for the
generic reference harness.
Field-level evidence classification and primary sources are in
[`configs/training/provenance.json`](../configs/training/provenance.json).

## Model and optimizer

| Field | Value |
|---|---:|
| Layers / hidden / FFN | 32 / 4096 / 21,504 |
| Attention heads / query groups | 32 / 8 |
| Activation / normalization | xIELU / RMSNorm |
| QK-LayerNorm | enabled |
| Embeddings | input/output untied |
| Linear bias | disabled |
| Sequence length | 4096 |
| Optimizer | AdEMAMix |
| β1 / β2 / β3 / α | 0.9 / 0.999 / 0.999 / 4.0 |
| β3 and α ramps | full 19,248 updates |
| Weight decay / gradient clip | 0.1 / 0.1 |
| Precision | bf16 parameters, fp32 main gradients |
| NaN/Inf checks | enabled |
| Loss | Goldfish, k=50, h=50 |

The previous public values β2=0.995 and a 13.5B-token horizon are obsolete.
Warmup remains the experimentally selected fixed 400 updates; it is not
recomputed as `2/(1-β2)`.

## Learning rate

| Field | Value |
|---|---:|
| Schedule | WSD |
| Peak | `5.5e-5` |
| Warmup initial | `5.5e-6` |
| Warmup | 400 updates / 409,600 sequences |
| Stable through | update 15,398 |
| Cooldown | 3,850 updates / 3,942,400 sequences |
| Cooldown shape | `1-sqrt` |
| Final | `5.5e-6` = 10% of peak |

WSD-10 is the selected baseline for this full D0 run. A different 10–30%
floor would be a separate LR experiment and must not be changed silently.

## Batch and parallelism

| Field | Value |
|---|---:|
| Microbatch | 2 sequences |
| Global batch | 1,024 sequences / 4,194,304 token slots |
| Updates | 19,248 |
| Training sequences | 19,709,952 |
| Active tokens | 80,729,939,067 |
| Token slots | 80,731,963,392 |
| Loss-inactive terminal slots | 2,024,325 |
| TP / PP / CP / DP | 2 / 1 / 1 / 32 |
| World size | 64 GPUs on 16 four-GPU nodes |
| Gradient accumulation | 16 |

## Corrected RoPE geometry

Use main-pretraining geometry, not the released post-long-context model
configuration:

```text
--max-position-embeddings 4096
--position-embedding-type rope
--rotary-base 500000
--use-rope-scaling
--rope-scaling-factor 8.0
```

## Data semantics and seeds

- D0 stationary windowed randomization from
  [`data_mix.d0.json`](../configs/training/data_mix.d0.json).
- Pool-permutation/mix seed: `20260801`.
- Megatron training/RNG seed: `20260609`.
- Reset attention masks and position IDs at document boundaries.
- Mask EOD targets from loss.
- Checkpoint averaging is disabled.

## Evaluation

The run evaluates 13 source-conditioned panels every 25 updates: HPLT,
non-HPLT, OpenArchives, Greek PhD, historical polytonic, English, German,
Russian, Chinese, code, math, Old Greek and neutral external Modern Greek.
Metrics include NLL, BPB and base-target versus added-target NLL.

Native GreekMMLU is evaluated at initialization, after warmup, approximately
every 5B tokens, cooldown start and final—20 milestones in total. Report full
and decontaminated subsets, zero-shot accuracy, choice NLL and correct-answer
BPB. Per-document validation runs at updates 0, 15,398 and 19,248.
