# Production tokenizer extension

Load the production tokenizer from
[`fffoivos/apertus-tokenizer-extension@fcd33ec`, subfolder
`greek-modern-polytonic-tokenizer`](https://huggingface.co/fffoivos/apertus-tokenizer-extension/tree/fcd33ec09fb7d86bc072b3a4b3e890efa6473b66/greek-modern-polytonic-tokenizer).

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "fffoivos/apertus-tokenizer-extension",
    revision="fcd33ec09fb7d86bc072b3a4b3e890efa6473b66",
    subfolder="greek-modern-polytonic-tokenizer",
    trust_remote_code=True,
)
assert len(tokenizer) == 148_992
```

The machine contract is
[`configs/tokenizer/extension.json`](../configs/tokenizer/extension.json).

## Geometry

| Stage | IDs | Count |
|---|---:|---:|
| Apertus base | `0..131071` | 131,072 |
| Modern Greek | `131072..148479` | 17,408 |
| Polytonic Greek | `148480..148991` | 512 |
| **Total** | `0..148991` | **148,992** |

`148992 = 582 × 256 = 291 × 512`. IDs and merges are contiguous and the
vocabulary requires zero dummy or padding entries with TP=2 and
`--make-vocab-size-divisible-by 256`.

The frozen `tokenizer.json` SHA-256 is:

```text
bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b
```

The published release includes the manifest, release audit, cutoff selection
and suspicious-token review. The polytonic block was appended sequentially so
every merge dependency already exists when that merge is applied.

One release-manifest field is stale: `dataset_tokenization_status` still says
`not_started`, although production-tokenizer corpus binaries now exist on
CSCS. This does not affect tokenizer files, IDs, merges or hashes, but the
metadata should be corrected in a future tokenizer-repository revision.

## Reconstruction tools

The scripts in [`scripts/tokenizer/`](../scripts/tokenizer) reconstruct the
cleaned 17,408-unit modern stage from upstream experiment artifacts. They do
not by themselves rebuild the second 512-unit polytonic stage; the published
SHA-pinned tokenizer is the production source of truth.

The removal policy remains
[`configs/tokenizer/removal_policy.json`](../configs/tokenizer/removal_policy.json).
It governs artifact removal in the modern stage and must not be reapplied to
renumber an already published tokenizer.
