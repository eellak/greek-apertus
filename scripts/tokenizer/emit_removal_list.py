#!/usr/bin/env python3
"""Emit bad-added-token removal manifests from an added-token glossary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REMOVE_LATIN_FRAGMENT_TAGS = {"-missing", "-decoded"}
REMOVE_LATIN_ACRONYM_LINENEW_FRAGS = {"LIN", "ENEW", "LINENEW"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glossary-jsonl", required=True, type=Path)
    parser.add_argument("--classified-jsonl", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--base-vocab-size", type=int, default=131_072)
    parser.add_argument("--target-added-units", type=int, default=17_408)
    parser.add_argument("--policy-version", default="2026-05-17b")
    return parser.parse_args()


def load_jsonl_by_id(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["id"])] = row
    return rows


def decide(glossary_row: dict) -> tuple[bool, str | None]:
    category = glossary_row.get("category")
    decoded = glossary_row.get("decoded") or ""

    if category == "mojibake":
        return True, "latin1_utf8_mojibake"
    if category == "mixed_script_token":
        return True, "mixed_script_artifact"
    if category == "postscript_glyph":
        return True, "pdf_postscript_glyph"
    if category == "code_identifier":
        return True, "cleaner_linenewline_placeholder"
    if category == "latin_acronym" and decoded in REMOVE_LATIN_ACRONYM_LINENEW_FRAGS:
        return True, "cleaner_linenewline_bpe_fragment"
    if category == "latin_fragment" and decoded in REMOVE_LATIN_FRAGMENT_TAGS:
        return True, "cleaner_extraction_tag"
    return False, None


def main() -> None:
    args = parse_args()
    glossary = load_jsonl_by_id(args.glossary_jsonl)
    classified = load_jsonl_by_id(args.classified_jsonl) if args.classified_jsonl else {}

    removals = []
    keeps = 0
    for token_id in sorted(glossary):
        row = glossary[token_id]
        remove, removal_class = decide(row)
        if not remove:
            keeps += 1
            continue
        classified_row = classified.get(token_id, {})
        removals.append(
            {
                "id": token_id,
                "decoded": row.get("decoded"),
                "category": row.get("category"),
                "lang_bucket": classified_row.get("lang_bucket") or row.get("lang_bucket"),
                "removal_class": removal_class,
                "meaning_snippet": (row.get("meaning") or row.get("meaning_snippet") or "")[:160],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    removal_path = args.out_dir / "removal_list.jsonl"
    with removal_path.open("w", encoding="utf-8") as handle:
        for row in removals:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_class = Counter(row["removal_class"] for row in removals)
    target_upper = args.base_vocab_size + args.target_added_units
    target_removals = [row for row in removals if int(row["id"]) < target_upper]
    summary = {
        "policy_version": args.policy_version,
        "base_vocab_size": args.base_vocab_size,
        "target_added_units": args.target_added_units,
        "target_total_vocab_size": target_upper,
        "removable_total": len(removals),
        "keepable_total": keeps,
        "removable_by_class": dict(sorted(by_class.items())),
        "removable_in_target": len(target_removals),
        "rules": [
            {"class": "latin1_utf8_mojibake", "predicate": "glossary.category == 'mojibake'"},
            {"class": "mixed_script_artifact", "predicate": "glossary.category == 'mixed_script_token'"},
            {"class": "pdf_postscript_glyph", "predicate": "glossary.category == 'postscript_glyph'"},
            {"class": "cleaner_linenewline_placeholder", "predicate": "glossary.category == 'code_identifier'"},
            {"class": "cleaner_linenewline_bpe_fragment", "predicate": "glossary.category == 'latin_acronym' and decoded in {'LIN','ENEW','LINENEW'}"},
            {"class": "cleaner_extraction_tag", "predicate": "glossary.category == 'latin_fragment' and decoded in {'-missing','-decoded'}"},
        ],
    }
    summary_path = args.out_dir / "decision_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {removal_path} ({len(removals)} rows)")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
