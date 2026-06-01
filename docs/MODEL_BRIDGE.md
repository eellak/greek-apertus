# Model Bridge

Apertus is released in Hugging Face format; the training path here uses Megatron-LM-Swiss-AI checkpoints. The bridge scripts make the round trip explicit.

## Scripts

- [`scripts/model_bridge/loader_apertus_hf.py`](../scripts/model_bridge/loader_apertus_hf.py) — HF → Megatron loader for standard tensors.
- [`scripts/model_bridge/patch_apertus_extras.py`](../scripts/model_bridge/patch_apertus_extras.py) — restores Apertus-specific xIELU and QK-Norm tensors after conversion.
- [`scripts/model_bridge/verify_hf_roundtrip.py`](../scripts/model_bridge/verify_hf_roundtrip.py) — independent HF checkpoint comparison, with optional prompt-logit checks.
- [`scripts/runtime/pretrain_gpt_te_guard.py`](../scripts/runtime/pretrain_gpt_te_guard.py) — runtime guard used by the training launcher.

## Fidelity

The loader emits only tensors that Megatron `saver_core.py` can accept. Apertus-specific tensors are patched back in with `patch_apertus_extras.py` and verified with `verify_hf_roundtrip.py`.

```bash
python3 tools/checkpoint/convert.py --model-type GPT --loader apertus_hf --saver core ...
python3 scripts/model_bridge/patch_apertus_extras.py --hf-dir "$HF_DIR" --megatron-dir "$CONVERTED" --out-dir "$PATCHED"
python3 scripts/model_bridge/verify_hf_roundtrip.py --reference-hf-dir "$HF_DIR" --roundtrip-hf-dir "$ROUNDTRIP_HF" --require-r17-match
```

The exact Megatron checkout and environment are deployment-specific and belong in the run metadata.
