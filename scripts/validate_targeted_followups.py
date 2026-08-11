#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def main() -> int:
    path = ROOT / "configs/training/targeted_8b_followups.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value["schema_version"] == "greek_apertus_targeted_8b_followups_v1", "schema drift")
    release = value["source_release"]
    require(release["revision"] == "987b8955fcd395c6219e39df9e64715457f69065", "release revision drift")
    require(release["anonymized_to_apertus_standard"] is True, "anonymization drift")
    require(release["additional_global_deduplication"] is False, "second dedup is forbidden")
    heldout = value["heldout_exclusion"]
    require(heldout["validation_panels"] == 13, "validation panel count drift")
    require(heldout["comparison"] == "exact_utf8_text_sha256", "heldout comparison drift")
    require(heldout["exclude_matches_before_packing"] is True, "heldout exclusion disabled")
    require(heldout["zero_overlap_postscan_required"] is True, "heldout postscan disabled")
    require(heldout["deduplication"] is False, "heldout exclusion must not deduplicate")
    shared = value["shared_scientific_contract"]
    require(shared["modern_foreign_old_greek_mix"] == [79, 20, 1], "mix drift")
    require(shared["execution_profile"] == "dp32_16node" and shared["dp64_allowed"] is False, "execution profile drift")
    a, b = value["experiment_a"], value["experiment_b"]
    require(a["modern_stream"]["openarchives_gr"]["training_tokens_pre_decontamination"] + a["modern_stream"]["greek_phd"]["training_tokens_pre_decontamination"] == 10_013_712_347, "academic total drift")
    require(a["planning_updates"] == 6092 and a["planning_normal_allocations"] == 2, "A planning drift")
    require(b["parent_checkpoint_update"] == 9536, "B checkpoint drift")
    require(b["remaining_non_hplt_active_tokens"] == 9_123_187_023, "B unseen mass drift")
    require(b["lr"]["load_optimizer_rng_and_sample_cursor"] is True, "B resume-state drift")
    require(b["planning_continuation_updates"] == 2754 and b["planning_final_absolute_update"] == 12290, "B geometry drift")
    print(json.dumps({"ok": True, "experiments": [a["id"], b["id"]], "launch_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
