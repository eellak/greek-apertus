#!/usr/bin/env python3
"""Summarize added-token glossary distributions across cutoff values."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glossary-jsonl", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--base-vocab-size", type=int, default=131_072)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[n * 1024 for n in range(1, 26)])
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return sorted(rows, key=lambda row: int(row["id"]))


def confidence_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.5:
        return "<0.5"
    if value < 0.7:
        return "0.5-0.7"
    if value < 0.9:
        return "0.7-0.9"
    return ">=0.9"


def sort_counter(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def distribution(rows: list[dict]) -> dict:
    by_category: Counter = Counter()
    by_language: Counter = Counter()
    by_struct: Counter = Counter()
    by_lex: Counter = Counter()
    by_conf: Counter = Counter()

    for row in rows:
        by_category[row.get("category", "unknown")] += 1
        by_language[row.get("language") or row.get("lang_bucket") or "unknown"] += 1
        morphology = row.get("greek_morphology") or {}
        if morphology.get("structure"):
            by_struct[morphology["structure"]] += 1
        if morphology.get("lexical"):
            by_lex[morphology["lexical"]] += 1
        by_conf[confidence_bucket(row.get("confidence"))] += 1

    return {
        "total": len(rows),
        "by_category": sort_counter(by_category),
        "by_language": sort_counter(by_language),
        "by_greek_structure": sort_counter(by_struct),
        "by_greek_lexical": sort_counter(by_lex),
        "confidence_buckets": sort_counter(by_conf),
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.glossary_jsonl)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_cutoff = {}
    for cutoff in sorted(set(args.cutoffs)):
        upper = args.base_vocab_size + cutoff
        sliced = [row for row in rows if int(row["id"]) < upper]
        if len(sliced) != cutoff:
            raise RuntimeError(f"cutoff {cutoff}: expected {cutoff} rows, got {len(sliced)}")
        payload = {
            "cutoff_added_units": cutoff,
            "total_vocab_size": upper,
            "id_range": [args.base_vocab_size, upper - 1],
            **distribution(sliced),
        }
        per_cutoff[str(cutoff)] = payload
        (args.out_dir / f"distribution_at_{cutoff}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "base_vocab_size": args.base_vocab_size,
        "cutoffs": sorted(set(args.cutoffs)),
        "per_cutoff": per_cutoff,
    }
    summary_path = args.out_dir / "cutoff_grid_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
