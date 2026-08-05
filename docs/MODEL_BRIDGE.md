# Model Bridge

Apertus is released in Hugging Face format; the training path here uses Megatron-LM-Swiss-AI checkpoints. The bridge scripts convert HF -> Megatron for training and back for verification.

## Scripts

- [`scripts/model_bridge/loader_apertus_hf.py`](../scripts/model_bridge/loader_apertus_hf.py) — HF → Megatron loader for standard tensors.
- [`scripts/model_bridge/patch_apertus_extras.py`](../scripts/model_bridge/patch_apertus_extras.py) — restores Apertus-specific xIELU and QK-Norm tensors after conversion.
- [`scripts/model_bridge/verify_hf_roundtrip.py`](../scripts/model_bridge/verify_hf_roundtrip.py) — independent HF checkpoint comparison, with optional prompt-logit checks.
- [`scripts/runtime/pretrain_gpt_te_guard.py`](../scripts/runtime/pretrain_gpt_te_guard.py) — runtime guard used by the training launcher.

## Fidelity

The loader emits only the tensors Megatron `saver_core.py` accepts, so conversion drops the Apertus-specific xIELU and QK-Norm tensors; `patch_apertus_extras.py` restores them and `verify_hf_roundtrip.py` confirms the round trip is lossless.

First install the loader into your `swiss-ai/Megatron-LM` checkout — it must sit in `tools/checkpoint/` alongside the upstream `saver_core.py` / `saver_swissai_hf.py`, which ship with that clone, not here:

```bash
ln -sf "$PWD/scripts/model_bridge/loader_apertus_hf.py" \
  "$MEGATRON_LM_DIR/tools/checkpoint/loader_apertus_hf.py"

# HF -> Megatron. --loader-transformer-impl transformer_engine is required:
# Apertus carries qknorm_impl=apex, which otherwise trips a saver_core assertion.
# --bf16 is a loader flag (convert.py's top-level parser does not define it).
python3 "$MEGATRON_LM_DIR/tools/checkpoint/convert.py" \
  --model-type GPT --loader apertus_hf --saver core \
  --load-dir "$HF_DIR" --save-dir "$CONVERTED" \
  --tokenizer-model "$HF_DIR" --bf16 \
  --loader-transformer-impl transformer_engine

# Restore the Apertus-specific xIELU / QK-Norm tensors raw conversion drops:
python3 scripts/model_bridge/patch_apertus_extras.py --hf-dir "$HF_DIR" --megatron-dir "$CONVERTED" --out-dir "$PATCHED"

# Verify the round trip (require zero standard / R17 / xIELU / QK / logit drift):
python3 scripts/model_bridge/verify_hf_roundtrip.py --reference-hf-dir "$HF_DIR" --roundtrip-hf-dir "$ROUNDTRIP_HF" --require-r17-match
```

The production run pins upstream Megatron commit
`c92402e39ef3c8e69ea378a59e79059dc14541f4` and the runtime patches recorded in
[`full_8b_mixed_cpt.json`](../configs/training/full_8b_mixed_cpt.json). A
different checkout invalidates the production conversion/runtime receipt.

The verified Token-Distillation checkpoint and exact verification-file hashes
are recorded in
[`token_distillation_8b.json`](../configs/initialization/token_distillation_8b.json).
The bridge scripts reference upstream files such as `saver_core.py` and
`saver_swissai_hf.py`; those ship with the Swiss-AI Megatron clone, not here.
