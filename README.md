# Greek Apertus

Public scientific contract and reusable tooling for continued pretraining of
[Apertus-8B](https://huggingface.co/swiss-ai/Apertus-8B-2509) on the complete
eligible GlossAPI Greek corpus with controlled replay.

## Full-run readiness review

| Component | Review result | Authority |
|---|---|---|
| Greek training corpus | Correct source dataset and exact post-exclusion HPLT/non-HPLT identities and quotas are frozen. `libduth` is included by owner directive, with its conflicting CC BY-NC-ND adjudication retained as an unresolved gate. | [`data_mix.d0.json`](configs/training/data_mix.d0.json) |
| Replay | Sufficient no-replacement foreign and Old-Greek capacity is available. The evidence boundary is explicit: source-family/overlap evidence is not an exact original-consumed-document manifest. | [`TRAINING_MIX.md`](docs/TRAINING_MIX.md) |
| Final 8B packed dataset | **Not yet complete.** The production-tokenizer pool, pack and D0 schedule receipts remain launch gates. | [`full_8b_mixed_cpt.json`](configs/training/full_8b_mixed_cpt.json) |
| Tokenizer | Ready and published: 131,072 base + 17,408 modern + 512 polytonic = 148,992 contiguous tokens, no padding, SHA-pinned. | [`extension.json`](configs/tokenizer/extension.json) |
| 8B embedding initialization | Ready: untied layer-11 Token Distillation plus separate output-row calibration, preservation checks and zero-drift HF/Megatron round trip passed. | [`token_distillation_8b.json`](configs/initialization/token_distillation_8b.json) |
| Training settings | Frozen for the D0 full run, including corrected RoPE, AdEMAMix, WSD-10, Goldfish, batch/parallelism and NaN/Inf checks. | [`full_8b_mixed_cpt.json`](configs/training/full_8b_mixed_cpt.json) |
| Evaluation | 13 source-conditioned panels, native GreekMMLU at 20 milestones, and per-document validation at initialization/cooldown/final are specified. | [`HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md) |
| Production launch | **Owner-authorized, technical gates pending.** D0 point-estimate acceptance and the `libduth` risk decision are recorded. Data receipts, the DP32/DP64 benchmark, initial evaluations, smokes, storage and scheduler gates must still pass. | [`owner_decisions_20260805.json`](configs/training/owner_decisions_20260805.json) |

The portable contract is validated by:

```bash
python3 scripts/validate_full_8b_contract.py
```

## Important execution boundary

The scripts already present in this repository remain useful bridge and
diagnostic references. The receipt-producing full-D0 scheduler and segmented
CSCS campaign are maintained in
[`fffoivos/train-apertus-with-glossapi`](https://github.com/fffoivos/train-apertus-with-glossapi),
under `subprojects/07_full_8b_cpt`. They must match the public JSON contract in
this repository before launch. The generic single-prefix trainer here is not a
substitute for the exact D0 schedule reader.

## Documentation

- [`TRAINING_MIX.md`](docs/TRAINING_MIX.md) — exact HPLT, GlossAPI and replay mix, provenance and receipt gates.
- [`TOKENIZER_EXTENSION.md`](docs/TOKENIZER_EXTENSION.md) — both extension stages and release hash.
- [`TOKEN_DISTILLATION.md`](docs/TOKEN_DISTILLATION.md) — exact untied 8B initialization procedure and evidence.
- [`HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md) — production training and evaluation settings.
- [`provenance.json`](configs/training/provenance.json) — field-by-field inherited, experimentally selected and derived settings.
- [`MODEL_BRIDGE.md`](docs/MODEL_BRIDGE.md) — HF/Megatron conversion fidelity.
- [`TRAINING.md`](docs/TRAINING.md) — execution boundary and launch gates.

## License

[MIT](LICENSE) — © 2026 GFOSS – Open Technologies Alliance.
