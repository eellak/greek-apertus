#!/usr/bin/env python3
"""Build append-only tokenizer cutoff variants.

The script takes an Apertus base tokenizer and a full continuous-BPE extension,
then writes variants that keep the base vocabulary plus the first N added ids.

It intentionally does not remove bad-token manifests. Curation should be applied
as a separate mask or as an explicit pruning step so the append-only id contract
is never broken by accident.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any


COPY_JSON_FILES = {
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-tokenizer-dir", required=True, type=Path)
    parser.add_argument("--full-tokenizer-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cutoffs", nargs="+", required=True, type=int)
    parser.add_argument("--base-vocab-size", type=int)
    parser.add_argument("--base-merge-count", type=int)
    parser.add_argument("--name-prefix", default="greek_apertus_added")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_tokenizer(path: Path) -> dict[str, Any]:
    tok_path = path / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"missing tokenizer.json: {tok_path}")
    return json.loads(tok_path.read_text(encoding="utf-8"))


def model(tok: dict[str, Any]) -> dict[str, Any]:
    value = tok.get("model")
    if not isinstance(value, dict):
        raise ValueError("tokenizer.json has no object-valued model field")
    if value.get("type") != "BPE":
        raise ValueError(f"expected BPE tokenizer model, got {value.get('type')!r}")
    return value


def vocab(model_obj: dict[str, Any]) -> dict[str, int]:
    value = model_obj.get("vocab")
    if not isinstance(value, dict):
        raise ValueError("tokenizer model has no vocab object")
    return {str(k): int(v) for k, v in value.items()}


def merges(model_obj: dict[str, Any]) -> list[Any]:
    value = model_obj.get("merges")
    if not isinstance(value, list):
        raise ValueError("tokenizer model has no merges list")
    return value


def infer_vocab_size(vocab_map: dict[str, int]) -> int:
    ids = sorted(vocab_map.values())
    if not ids:
        raise ValueError("empty vocabulary")
    expected = list(range(ids[-1] + 1))
    if ids != expected:
        raise ValueError("vocabulary ids are not contiguous; pass an explicit implementation plan")
    return ids[-1] + 1


def copy_sidecars(src_dir: Path, dst_dir: Path) -> None:
    for name in COPY_JSON_FILES:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def write_variant(
    base_dir: Path,
    full_tok: dict[str, Any],
    full_vocab: dict[str, int],
    full_merges: list[Any],
    out_dir: Path,
    name_prefix: str,
    cutoff: int,
    base_vocab_size: int,
    base_merge_count: int,
    overwrite: bool,
) -> Path:
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if cutoff % 128 != 0:
        raise ValueError(f"cutoff {cutoff} is not 128-aligned")

    target_vocab_size = base_vocab_size + cutoff
    target_merge_count = base_merge_count + cutoff
    if target_vocab_size > infer_vocab_size(full_vocab):
        raise ValueError(f"cutoff {cutoff} exceeds full tokenizer vocab")
    if target_merge_count > len(full_merges):
        raise ValueError(f"cutoff {cutoff} exceeds full tokenizer merge list")

    dst = out_dir / f"{name_prefix}_{cutoff}"
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    variant = copy.deepcopy(full_tok)
    variant_model = model(variant)
    variant_model["vocab"] = {
        token: tok_id
        for token, tok_id in sorted(full_vocab.items(), key=lambda item: item[1])
        if tok_id < target_vocab_size
    }
    variant_model["merges"] = full_merges[:target_merge_count]

    added_tokens = variant.get("added_tokens")
    if isinstance(added_tokens, list):
        variant["added_tokens"] = [
            row for row in added_tokens
            if not isinstance(row, dict) or int(row.get("id", -1)) < target_vocab_size
        ]

    (dst / "tokenizer.json").write_text(
        json.dumps(variant, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    copy_sidecars(base_dir, dst)
    metadata = {
        "base_vocab_size": base_vocab_size,
        "added_units": cutoff,
        "total_vocab_size": target_vocab_size,
        "base_merge_count": base_merge_count,
        "total_merge_count": target_merge_count,
        "append_only": True,
    }
    (dst / "cutoff_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return dst


def main() -> None:
    args = parse_args()
    base_tok = read_tokenizer(args.base_tokenizer_dir)
    full_tok = read_tokenizer(args.full_tokenizer_dir)

    base_model = model(base_tok)
    full_model = model(full_tok)
    base_vocab = vocab(base_model)
    full_vocab = vocab(full_model)
    full_merges = merges(full_model)

    base_vocab_size = args.base_vocab_size or infer_vocab_size(base_vocab)
    base_merge_count = args.base_merge_count or len(merges(base_model))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cutoff in sorted(set(args.cutoffs)):
        dst = write_variant(
            args.base_tokenizer_dir,
            full_tok,
            full_vocab,
            full_merges,
            args.out_dir,
            args.name_prefix,
            cutoff,
            base_vocab_size,
            base_merge_count,
            args.overwrite,
        )
        print(f"built {dst}")


if __name__ == "__main__":
    main()
