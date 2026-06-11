# Tokenizer Extension

Two utilities:

1. Build the cleaned 17,408-unit append-only Greek extension.
2. Emit a removal manifest for added tokens that are extraction, encoding, or cleaner-residue artifacts (see `removal_policy.json` for the six named classes).

Current extension contract: [`configs/tokenizer/extension.json`](../configs/tokenizer/extension.json). Current policy: [`configs/tokenizer/removal_policy.json`](../configs/tokenizer/removal_policy.json).

## Append-Only Contract

The base Apertus ids stay fixed. The modern Greek extension adds 17,408 cleaned BPE units, giving a total vocabulary size of 148,480 (`256 x 580`). The added id range is `131072 .. 148479`.

The 17,408 cutoff is a frozen, empirically-chosen decision, not a free knob: it was selected at the fertility / token-firing knee via the swiss-ai TokEval intrinsic-evaluation sweep over a 1k-spaced grid of candidate cutoffs (an earlier analytic anchor of 11,264 was superseded). `build_clean_extension.py` exposes `--added-units`, but the shipped value is fixed.

## Removal Policy

Tokens matching any of the six removal classes are structurally excluded from the shipped extension. The builder walks the full continuous-BPE extension, skips ids in the removal manifest, and backfills with the next valid merges so the final added block remains contiguous and 256-aligned.

## Commands

These commands are reference recipes: the annotated inputs they consume
(`added_token_glossary.jsonl`, `classified_added_tokens.jsonl`, and the uncurated
full continuous-BPE Greek extension) are produced upstream in the
tokenizer-experiment pipeline and are not shipped here.

The two scripts form one pipeline: `emit_removal_list.py` runs first and writes
`removal_list.jsonl`, which `build_clean_extension.py` then consumes via
`--removal-list`.

Emit the removal manifest — per-added-token keep/remove decisions, derived from
the upstream glossary and per-token classification:

```bash
python3 scripts/tokenizer/emit_removal_list.py \
  --glossary-jsonl /path/to/added_token_glossary.jsonl \
  --classified-jsonl /path/to/classified_added_tokens.jsonl \
  --out-dir /path/to/manifests
```

Build the cleaned extension — `--full-tokenizer-dir` is the uncurated
continuous-BPE Greek extension before curation; the builder skips the removed
ids and backfills with the next valid merges:

```bash
python3 scripts/tokenizer/build_clean_extension.py \
  --base-tokenizer-dir /path/to/apertus-base-tokenizer \
  --full-tokenizer-dir /path/to/full-extension-tokenizer \
  --removal-list /path/to/manifests/removal_list.jsonl \
  --out-dir /path/to/tokenizers
```
