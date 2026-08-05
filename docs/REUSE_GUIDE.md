# Reuse guide

For the selected full 8B D0 run, preserve these together:

- [`extension.json`](../configs/tokenizer/extension.json)
- [`token_distillation_8b.json`](../configs/initialization/token_distillation_8b.json)
- [`data_mix.d0.json`](../configs/training/data_mix.d0.json)
- [`full_8b_mixed_cpt.json`](../configs/training/full_8b_mixed_cpt.json)

Changing tokenizer revision, embedding initialization, document identities,
pool quotas/order, LR floor, optimizer, batch geometry, RoPE, loss or evaluation
cadence changes the scientific run and requires a new recipe ID.

Operationally tunable choices are node count, segment walltime, scheduler
submission time and transport tuning, but only after an exact-shape benchmark
shows that the scientific batch and sample order remain unchanged. The current
proven production shape is 16 nodes / 64 GPUs / TP2 / DP32.

The training seed `20260609` and D0 mix seed `20260801` serve different
purposes. Do not collapse them into one implicit sampler seed.

The generic scripts in this repository are reusable diagnostics and bridge
tools. The production scheduler is the receipt-bound orchestration in
`fffoivos/train-apertus-with-glossapi/subprojects/07_full_8b_cpt`.
