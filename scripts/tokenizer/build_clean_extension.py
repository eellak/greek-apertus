#!/usr/bin/env python3
"""Build the cleaned 17,408-unit Greek Apertus tokenizer extension."""

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
    parser.add_argument("--removal-list", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--name", default="apertus_greek_modern_148480")
    parser.add_argument("--added-units", type=int, default=17_408)
    parser.add_argument("--base-vocab-size", type=int)
    parser.add_argument("--base-merge-count", type=int)
    parser.add_argument("--alignment", type=int, default=256)
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
    if ids != list(range(ids[-1] + 1)):
        raise ValueError("vocabulary ids are not contiguous")
    return ids[-1] + 1


def merge_parts(row: Any) -> tuple[str, str]:
    if isinstance(row, str):
        parts = row.split(" ", 1)
    elif isinstance(row, list) and len(row) == 2:
        parts = [str(row[0]), str(row[1])]
    else:
        raise ValueError(f"unsupported merge row: {row!r}")
    if len(parts) != 2:
        raise ValueError(f"unsupported merge row: {row!r}")
    return parts[0], parts[1]


def load_removal_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.add(int(row["id"]))
    return ids


def copy_sidecars(src_dir: Path, dst_dir: Path) -> None:
    for name in COPY_JSON_FILES:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def build_extension(
    base_tok: dict[str, Any],
    full_tok: dict[str, Any],
    removal_ids: set[int],
    added_units: int,
    base_vocab_size: int,
    base_merge_count: int,
    alignment: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if added_units <= 0:
        raise ValueError(f"added_units must be positive, got {added_units}")
    target_vocab_size = base_vocab_size + added_units
    if base_vocab_size % alignment != 0:
        raise ValueError(f"base vocab size {base_vocab_size} is not {alignment}-aligned")
    if target_vocab_size % alignment != 0:
        raise ValueError(f"target vocab size {target_vocab_size} is not {alignment}-aligned")

    base_model = model(base_tok)
    full_model = model(full_tok)
    base_vocab = vocab(base_model)
    full_vocab = vocab(full_model)
    full_merges = merges(full_model)
    token_by_full_id = {tok_id: token for token, tok_id in full_vocab.items()}

    for token, tok_id in base_vocab.items():
        if full_vocab.get(token) != tok_id:
            raise ValueError(f"base token id mismatch in full tokenizer: {token!r}")

    new_vocab = {
        token: tok_id
        for token, tok_id in sorted(base_vocab.items(), key=lambda item: item[1])
        if tok_id < base_vocab_size
    }
    new_merges = list(merges(base_model))
    skipped_tokens: set[str] = set()
    accepted = []
    skipped_for_policy = []
    skipped_for_dependency = []

    offset = 0
    while len(accepted) < added_units:
        old_id = base_vocab_size + offset
        merge_index = base_merge_count + offset
        if old_id not in token_by_full_id:
            raise ValueError("full tokenizer ended before enough clean added units were found")
        if merge_index >= len(full_merges):
            raise ValueError("full merge list ended before enough clean added units were found")

        merge = full_merges[merge_index]
        left, right = merge_parts(merge)
        token = token_by_full_id[old_id]
        if left + right != token:
            raise ValueError(
                f"merge/vocab mismatch at full id {old_id}: {left!r} + {right!r} != {token!r}"
            )

        if old_id in removal_ids:
            skipped_tokens.add(token)
            skipped_for_policy.append({"old_id": old_id, "token": token})
        elif left in skipped_tokens or right in skipped_tokens:
            skipped_tokens.add(token)
            skipped_for_dependency.append({"old_id": old_id, "token": token})
        else:
            new_id = base_vocab_size + len(accepted)
            new_vocab[token] = new_id
            new_merges.append(merge)
            accepted.append({"old_id": old_id, "new_id": new_id, "token": token})
        offset += 1

    variant = copy.deepcopy(full_tok)
    variant_model = model(variant)
    variant_model["vocab"] = new_vocab
    variant_model["merges"] = new_merges

    added_tokens = variant.get("added_tokens")
    if isinstance(added_tokens, list):
        variant["added_tokens"] = [
            row for row in added_tokens
            if not isinstance(row, dict) or int(row.get("id", -1)) < base_vocab_size
        ]

    metadata = {
        "base_vocab_size": base_vocab_size,
        "added_units": added_units,
        "total_vocab_size": target_vocab_size,
        "vocab_alignment": alignment,
        "append_only_base_ids": True,
        "construction": "skip_removed_tokens_and_backfill",
        "full_added_units_walked": offset,
        "removed_in_target_scope": len(skipped_for_policy),
        "dependency_skips": len(skipped_for_dependency),
        "accepted_first_old_id": accepted[0]["old_id"],
        "accepted_last_old_id": accepted[-1]["old_id"],
    }
    return variant, metadata


def main() -> None:
    args = parse_args()
    base_tok = read_tokenizer(args.base_tokenizer_dir)
    full_tok = read_tokenizer(args.full_tokenizer_dir)
    removal_ids = load_removal_ids(args.removal_list)

    base_model = model(base_tok)
    base_vocab_size = args.base_vocab_size or infer_vocab_size(vocab(base_model))
    base_merge_count = args.base_merge_count or len(merges(base_model))

    variant, metadata = build_extension(
        base_tok,
        full_tok,
        removal_ids,
        args.added_units,
        base_vocab_size,
        base_merge_count,
        args.alignment,
    )

    dst = args.out_dir / args.name
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    (dst / "tokenizer.json").write_text(
        json.dumps(variant, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    copy_sidecars(args.base_tokenizer_dir, dst)
    (dst / "extension_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"built {dst}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
