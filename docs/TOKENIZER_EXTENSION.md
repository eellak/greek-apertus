# Tokenizer Extension

Two utilities:

1. Build append-only cutoff variants from a full continuous-BPE extension.
2. Emit a removal manifest for added tokens that are extraction or encoding artifacts.

Current policy: [`configs/tokenizer/removal_policy.json`](../configs/tokenizer/removal_policy.json). Current cutoff grid: [`configs/tokenizer/extension_cutoffs.json`](../configs/tokenizer/extension_cutoffs.json).

## Append-Only Contract

The base Apertus ids stay fixed. A cutoff variant keeps ids `0 .. BASE_VOCAB_SIZE + ADDED_UNITS - 1`, so base Apertus weights remain valid and the embedding table extends by appending rows.

## Removal Policy

Bad added tokens are written to a manifest rather than silently deleted by the cutoff builder. Removing tokens from a BPE vocabulary can renumber ids or create holes; the manifest lets downstream code mask, zero-init, or prune them deliberately.

## Commands

Build cutoff variants:

```bash
python3 scripts/tokenizer/build_cutoff_variants.py \
  --base-tokenizer-dir /path/to/apertus-base-tokenizer \
  --full-tokenizer-dir /path/to/full-extension-tokenizer \
  --out-dir /path/to/cutoff-tokenizers \
  --cutoffs 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264
```

Emit removal manifests:

```bash
python3 scripts/tokenizer/emit_removal_list.py \
  --glossary-jsonl /path/to/added_token_glossary.jsonl \
  --classified-jsonl /path/to/classified_added_tokens.jsonl \
  --out-dir /path/to/manifests
```
