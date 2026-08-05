# Full 8B training mix

The selected full-run arm is `D0_mixed`: HPLT and all eligible non-HPLT
GlossAPI documents are randomized together in their natural post-exclusion
proportions. Foreign replay remains 20% and Old-Greek replay remains 1% at all
points. There is no internal GlossAPI curriculum.

The machine-readable authority is
[`configs/training/data_mix.d0.json`](../configs/training/data_mix.d0.json).

## Exact active-token quotas

| Pool | Active tokens | Share of total |
|---|---:|---:|
| HPLT Modern Greek | 44,042,201,419 | 54.554979% |
| GlossAPI/non-HPLT Modern Greek | 19,734,450,444 | 24.445021% |
| Foreign replay | 16,145,987,813 | 20.000000% |
| Old-Greek replay | 807,299,391 | 1.000000% |
| **Total** | **80,729,939,067** | **100%** |

Within the 63,776,651,863-token Modern-Greek stream, HPLT is
69.0569356% and GlossAPI/non-HPLT is 30.9430644%.

The 19,248 updates contain 80,731,963,392 token slots. Exactly 2,024,325
terminal slots are loss-inactive filler; they are not training tokens.

## Modern-Greek source

The source is the complete eligible pass of
[`fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2` at revision
`3f97cec`](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2/tree/3f97cec48af502f4996cf8ff20b02660e2dd3d31).
`source_dataset` and `source_doc_id` preserve document identity and source
metadata. HPLT is selected by `source_dataset =~ ^HPLT/`; the non-HPLT pool is
every other eligible source.

`libduth` is technically included by the explicit owner directive and is
present in public v2. There is still a policy-evidence conflict: the older
technical license adjudication records CC BY-NC-ND and does not support
training or redistribution. This repository makes no legal conclusion from
that conflict. Before production use/publication, reconcile it with permission
evidence or record explicit risk acceptance.

Heldouts and GreekMMLU contamination matches are excluded before schedule
construction. A final global exact-content check over text SHA-256 removed one
four-token duplicate document from the Modern-Greek pool. This is exact
deduplication evidence, not a claim of global semantic or near-duplicate
removal.

## Replay claim boundary

Foreign replay is selected without replacement from the already-vetted public
source families FineWeb-Edu, FineWeb2-HQ/FineWeb2, FineMath and StarCoder.
Those are Apertus training source families. We do **not** claim that every
selected document can be proven to have appeared in the exact hidden
original-consumption manifest.

Old-Greek replay is selected without replacement from the document-level
Nanochat-overlap audit pool. This proves the overlap classification used by the
project; it is not an exact original-consumed-document manifest either.

Available capacities exceed the requested selections:

| Pool | Capacity tokens | Selected tokens |
|---|---:|---:|
| Foreign replay | 45,299,005,175 | 16,145,987,813 |
| Old-Greek replay | 2,666,110,500 | 807,299,391 |

Replay exact-content duplicates are measured and receipted but original
source-record multiplicity is preserved. That policy avoids silently changing
the replay distribution.

## Receipt status

The completed scheduling pool receipt on CSCS is:

```text
/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/dataset_scheduling_0p5b/
20260802T104539Z-mini-schedule-v1/pool_corpus_receipt.json
sha256 76658cc8495b58a3a3dadc8aca6d16c7fed627ef8bced89d17a877f9f9125014
```

It freezes the exact identities, counts, exclusions, replay capacity and the
12 inherited heldouts. It used the 0.5B-compatible tokenizer overlay carrying
the same appended Greek merge chain. The production-tokenizer source binaries
also exist, but the final 8B pool, packing and D0 schedule receipts have not yet
been generated. Those three receipts are mandatory before launch and must
reproduce the exact active-token accounting above.
