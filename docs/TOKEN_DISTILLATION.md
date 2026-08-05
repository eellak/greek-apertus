# Token Distillation initialization

The production 8B initialization is complete and verified. Apertus-8B has
untied input and output embeddings, so the procedure deliberately does more
than learn one shared table.

Machine-readable authority:
[`configs/initialization/token_distillation_8b.json`](../configs/initialization/token_distillation_8b.json).

## Procedure

1. Start from `swiss-ai/Apertus-8B-2509@3162c996...` and the published,
   **uncpt** 148,480-row layer-11 `TokenDistil-Init` checkpoint. Do not start
   from a CPT-trained checkpoint.
2. Preserve all base IDs and append the 512 polytonic IDs `148480..148991` in
   dependency-safe merge order.
3. Initialize the new input and output rows from their merge parents, with norm
   calibration against the modern added rows.
4. Train the new input rows with layer-11 Token Distillation: 25 snippets per
   token, one epoch, batch 8, LR `1e-4`, bf16.
5. Because embeddings are untied, calibrate the 512 output rows separately
   with next-token cross-entropy for 400 steps at sequence length 512, LR
   `2e-4`, gradient clip 1.0 and seed `20260729`.
6. Require exact preservation of every pre-existing input row, output row and
   non-embedding tensor, finite/nonzero new rows, and a zero-drift
   HF → Megatron TP=2 → HF round trip.

The fixed-ID adapter and production builder are preserved in
[`fffoivos/train-apertus-with-glossapi`](https://github.com/fffoivos/train-apertus-with-glossapi):

- [`train_retok_td.py`](https://github.com/fffoivos/train-apertus-with-glossapi/blob/86c1b8fe362233bba6e4e2ca92eb4535287fb240/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation/train_retok_td.py)
- [`build_production_init.sbatch`](https://github.com/fffoivos/train-apertus-with-glossapi/blob/86c1b8fe362233bba6e4e2ca92eb4535287fb240/subprojects/05_token_distillation_cpt/06_25b_midtraining_probe/initialization/build_production_init.sbatch)
- [`verify_production_init.py`](https://github.com/fffoivos/train-apertus-with-glossapi/blob/86c1b8fe362233bba6e4e2ca92eb4535287fb240/subprojects/05_token_distillation_cpt/06_25b_midtraining_probe/initialization/verify_production_init.py)

## Frozen evidence

```text
Production verification:
/capstor/scratch/cscs/fffoivos/models/greek-cpt25b-init/
20260731T124000Z-cpt25b-v1/production_init_verification.json
sha256 7b4adb065f401305064cda8c45cc1b2c1430366ef8fbf5ee060f154be5b48e26

Round-trip verification:
/capstor/scratch/cscs/fffoivos/models/greek-cpt25b-init-roundtrip/
20260731T124000Z-cpt25b-v1/work/verification.json
sha256 06ddc69f4787f470b85d4be55992f898b98d8ab4fd8ab34a89a4d3bb96eb7e33

Megatron TP=2 checkpoint:
/capstor/scratch/cscs/fffoivos/models/greek-cpt25b-init-roundtrip/
20260731T124000Z-cpt25b-v1/megatron_tp2_r17patched
```

These are file hashes. They must not be confused with semantic tensor-tree
digests that may also appear inside verification reports.
