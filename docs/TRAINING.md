# Training and launch boundary

The full-run scientific contract is frozen in
[`configs/training/full_8b_mixed_cpt.json`](../configs/training/full_8b_mixed_cpt.json).
The production campaign is not launch-ready until every receipt and smoke gate
below passes.

## Production orchestration

The exact schedule reader, receipt-producing dataset pipeline, six-segment
Clariden launcher, checkpoint gates, source validation, GreekMMLU conversion
and per-document evaluation are maintained in
[`fffoivos/train-apertus-with-glossapi`](https://github.com/fffoivos/train-apertus-with-glossapi)
under `subprojects/07_full_8b_cpt`.

This repository intentionally does not pretend that the older generic
single-prefix launcher can reproduce the receipt-bound D0 sequence. In
particular:

- [`train_apertus_cpt.sbatch`](../scripts/train/train_apertus_cpt.sbatch) is a
  diagnostic/reference command builder;
- [`submit_cpt_chain.sh`](../scripts/train/submit_cpt_chain.sh) is a generic
  equal-segment submitter;
- production uses exact schedule IDs, exact checkpoint iterations and a patched
  extra-validation runtime from the orchestration repository.

## Runtime contract

Production uses Swiss-AI Megatron upstream commit
`c92402e39ef3c8e69ea378a59e79059dc14541f4`, plus two hash-pinned patches:

| Patch | SHA-256 |
|---|---|
| Extra source-conditioned validation | `2e6810fa8b6c25597ccb3bcb9dc1ff5bf843ead2337e3edde0344605a23ec4c6` |
| Exact evaluation/checkpoint iterations | `6d9392cfb0dd08e62089d0a98e2817b222bb9a25ee5cefa8f3cdf29a8ce16bea` |

`f8d8a30ba22a807321ec5875abbd9692b9282940` appeared in a patch header. It is
not the checked-out runtime commit and must not be used as one. Every launch
must consume a runtime receipt proving upstream HEAD, patch hashes, changed
file inventory and code-tree hash.

The existing frozen runtime receipt is
`/iopsstor/scratch/cscs/fffoivos/orchestration/dataset-scheduling-0p5b/20260803T093500Z-megatron-production-c92402e-v1.receipt.json`
(file SHA-256 `99b9ecbd49bec162941d1b9bd11996b39ab421f4c43eca962c09ca37fdc7b36a`).
The production launcher revalidates it against the live checkout before every
segment.

## Clariden shape

- account `a0140`, partition `normal`;
- `pytorch/v2.9.1:v2`, view `default`;
- 16 nodes × 4 GH200 GPUs;
- TP=2, DP=32, PP=1, CP=1;
- six 12-hour-safe segments at updates
  `0, 3208, 6416, 9624, 12832, 16040, 19248`;
- full optimizer, RNG and sample-cursor parity required at every resume.

## Mandatory launch gates

1. Freeze a clean immutable orchestration checkout and patched Megatron runtime receipt.
2. Rebuild the production-8B pool, packed-data and D0 schedule receipts and match [`data_mix.d0.json`](../configs/training/data_mix.d0.json).
3. Freeze all 13 validation panels and GreekMMLU decontamination bindings.
4. Reverify tokenizer and Token-Distillation artifact hashes.
5. Run finite initial source validation, all initial per-document panels and initial full/clean GreekMMLU.
6. Pass a two-update train/save/resume smoke with optimizer, RNG and sample-cursor parity.
7. Pass Megatron-to-HF conversion and GreekMMLU evaluation equivalence smoke.
8. Verify at least 6 TB available for checkpoints and evaluation conversions.
9. Record a fresh scheduler/capacity snapshot.
10. Obtain explicit production launch authorization.

The production launcher is dry-run by default and must require the exact
confirmation string `APERTUS8B_FULL_MIXED_CPT` for a live submission.

## Generic diagnostic harness

For small non-production tests, copy
[`dataset_paths.example.env`](../configs/training/dataset_paths.example.env),
override the complete token/sample/update geometry together, and inspect the
dry run from [`submit_cpt_chain.sh`](../scripts/train/submit_cpt_chain.sh).
Do not describe results from this generic single-prefix path as the full D0
run.
