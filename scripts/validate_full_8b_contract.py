#!/usr/bin/env python3
"""Validate the portable full Apertus-8B D0 scientific contract."""

from __future__ import annotations

import json
import os
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def clean_env_values() -> dict[str, str]:
    keys = [
        "ADEMA_BETA2",
        "TRAIN_TOKENS",
        "TRAIN_TOKEN_SLOTS",
        "LOSS_INACTIVE_TAIL_SLOTS",
        "TRAIN_SAMPLES",
        "TRAIN_ITERS",
        "LR_WARMUP_ITERS",
        "LR_WSD_DECAY_SAMPLES",
        "CURRICULUM_ORDER_MODE",
        "DATA_SEED",
        "MIX_SEED",
        "ROTARY_BASE",
        "ROPE_SCALING_FACTOR",
    ]
    command = "source \"$1\"; " + "; ".join(
        f"printf '%s\\t%s\\n' {key} \"${{{key}}}\"" for key in keys
    )
    completed = subprocess.run(
        ["bash", "-c", command, "_", str(ROOT / "configs/training/cpt.env")],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    return dict(line.split("\t", 1) for line in completed.stdout.splitlines())


def main() -> int:
    tokenizer = load("configs/tokenizer/extension.json")
    init = load("configs/initialization/token_distillation_8b.json")
    mix = load("configs/training/data_mix.d0.json")
    recipe = load("configs/training/full_8b_mixed_cpt.json")
    regime = load("configs/training/cpt.regime.json")
    provenance = load("configs/training/provenance.json")

    require(tokenizer["base_vocab_size"] == 131_072, "base vocabulary drift")
    require(tokenizer["modern_added_units"] == 17_408, "modern extension drift")
    require(tokenizer["polytonic_added_units"] == 512, "polytonic extension drift")
    require(tokenizer["total_added_units"] == 17_920, "total extension drift")
    require(tokenizer["total_vocab_size"] == 148_992, "vocabulary size drift")
    require(tokenizer["total_vocab_size"] % 256 == 0, "256 alignment drift")
    require(tokenizer["total_vocab_size"] % 512 == 0, "512 alignment drift")
    require(tokenizer["padding_tokens"] == 0, "padding tokens are forbidden")
    tokenizer_sha = "bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b"
    require(tokenizer["source"]["tokenizer_json_sha256"] == tokenizer_sha, "tokenizer hash drift")
    require(tokenizer["upstream_manifest_metadata_issue"]["published_value"] == "not_started", "tokenizer manifest issue tracking drift")
    require(init["tokenizer"]["tokenizer_json_sha256"] == tokenizer_sha, "init/tokenizer binding drift")
    require(init["base_model"]["input_output_embeddings_tied"] is False, "8B embeddings must remain untied")
    require(init["modern_stage"]["target_layer"] == 11, "modern TD layer drift")
    require(init["polytonic_stage"]["input_token_distillation"]["target_layer"] == 11, "polytonic TD layer drift")
    require(
        init["verified_artifact"]["production_verification"]["file_sha256"]
        == "7b4adb065f401305064cda8c45cc1b2c1430366ef8fbf5ee060f154be5b48e26",
        "production init receipt hash drift",
    )
    require(
        init["verified_artifact"]["roundtrip_verification"]["file_sha256"]
        == "06ddc69f4787f470b85d4be55992f898b98d8ab4fd8ab34a89a4d3bb96eb7e33",
        "round-trip receipt hash drift",
    )

    tokens = mix["active_tokens"]
    require(tokens["hplt_modern_greek"] + tokens["glossapi_non_hplt_modern_greek"] == tokens["all_modern_greek"], "modern token sum drift")
    require(tokens["all_modern_greek"] + tokens["foreign_replay"] + tokens["old_greek_replay"] == tokens["total"], "total active-token sum drift")
    require(tokens["total"] == 80_729_939_067, "active-token horizon drift")
    require(mix["schedule"]["arm"] == "D0_mixed", "data-order arm drift")
    require(mix["schedule"]["seed"] == 20260801, "mix seed drift")
    require(mix["schedule"]["selected_without_replacement"] is True, "replacement sampling is forbidden")
    libduth = mix["modern_greek_source"]["libduth"]
    require(libduth["legal_conclusion_claimed_by_this_repository"] is False, "repository must not manufacture a libduth legal conclusion")
    require("libduth_permission_evidence_conflict_is_reconciled_or_explicitly_accepted" in mix["required_before_launch"], "libduth evidence-conflict gate missing")
    require(abs(Decimal(mix["fractions"]["modern_greek_of_total"]) - Decimal("0.79")) < Decimal("1e-11"), "modern fraction drift")
    slots = mix["slot_geometry"]
    require(slots["training_sequences"] == slots["optimizer_updates"] * slots["global_batch_sequences"], "training sequence horizon drift")
    require(slots["token_slots"] == slots["training_sequences"] * slots["sequence_length"], "token-slot geometry drift")
    require(slots["token_slots"] - slots["loss_active_tokens"] == slots["loss_inactive_tail_slots"], "inactive-tail accounting drift")

    model = recipe["model"]
    require((model["max_position_embeddings"], model["rope"]["base"], model["rope"]["use_scaling"], model["rope"]["scaling_factor"]) == (4096, 500000, True, 8.0), "corrected RoPE geometry drift")
    opt = recipe["optimization"]
    require((opt["beta1"], opt["beta2"], opt["beta3"], opt["alpha"]) == (0.9, 0.999, 0.999, 4.0), "optimizer drift")
    require(opt["nan_and_inf_checks_enabled"] is True, "NaN/Inf checks must be enabled")
    require(opt["learning_rate"]["warmup_updates"] == 400, "warmup drift")
    require(opt["learning_rate"]["final"] == opt["learning_rate"]["peak"] / 10, "WSD-10 floor drift")
    batch = recipe["batch_and_parallelism"]
    require(batch["training_updates"] == slots["optimizer_updates"], "recipe/mix update drift")
    require(batch["training_samples"] == slots["training_sequences"], "recipe/mix sample drift")
    require(batch["world_size"] == batch["nodes"] * batch["gpus_per_node"], "world-size drift")
    require(batch["data_parallel"] == batch["world_size"] // batch["tensor_parallel"], "data-parallel drift")
    require(batch["gradient_accumulation_steps"] == batch["global_batch_sequences"] // (batch["data_parallel"] * batch["micro_batch_sequences"]), "gradient accumulation drift")
    boundaries = recipe["segments"]["boundaries"]
    require(boundaries == [0, 3208, 6416, 9624, 12832, 16040, 19248], "segment boundary drift")
    checkpoints = set(recipe["evaluation"]["saved_checkpoint_updates"])
    greek = recipe["evaluation"]["greekmmlu"]["checkpoint_updates"]
    require(len(recipe["evaluation"]["source_conditioned"]["panels"]) == 13, "validation panel count drift")
    require(len(greek) == 20 and greek[0] == 0 and greek[-1] == 19248, "GreekMMLU cadence drift")
    require(not (set(greek[1:]) - checkpoints), "GreekMMLU milestone missing saved checkpoint")
    require(recipe["data_semantics"]["checkpoint_averaging"] is False, "checkpoint averaging was excluded")
    require("libduth_permission_evidence_conflict_reconciled_or_explicitly_accepted" in recipe["launch_gates"], "recipe libduth gate missing")

    software = recipe["software"]
    require(software["megatron_upstream_commit"] == "c92402e39ef3c8e69ea378a59e79059dc14541f4", "Megatron commit drift")
    patch_hashes = {row["name"]: row["sha256"] for row in software["required_runtime_patches"]}
    require(patch_hashes == {
        "extra_valid": "2e6810fa8b6c25597ccb3bcb9dc1ff5bf843ead2337e3edde0344605a23ec4c6",
        "exact_eval_iterations": "6d9392cfb0dd08e62089d0a98e2817b222bb9a25ee5cefa8f3cdf29a8ce16bea",
    }, "runtime patch drift")
    runtime = software["validated_runtime_receipt"]
    require(runtime["file_sha256"] == "99b9ecbd49bec162941d1b9bd11996b39ab421f4c43eca962c09ca37fdc7b36a", "runtime receipt hash drift")
    require(runtime["git_diff_sha256"] == "550a9f570a99f1ca20773bd2d97795210ce153ace1be794cf9f11aeb4b2238e6", "runtime diff hash drift")
    require(runtime["revalidate_at_every_job_start"] is True, "runtime revalidation must remain enabled")
    require(regime["authoritative_recipe"] == "configs/training/full_8b_mixed_cpt.json", "regime authority drift")
    require(recipe["provenance"] == "configs/training/provenance.json", "provenance pointer drift")
    require(provenance["recipe"] == "configs/training/full_8b_mixed_cpt.json", "provenance recipe binding drift")
    require("D0_mixed_order" in provenance["field_provenance"], "D0 provenance missing")

    env = clean_env_values()
    expected_env = {
        "ADEMA_BETA2": "0.999",
        "TRAIN_TOKENS": "80729939067",
        "TRAIN_TOKEN_SLOTS": "80731963392",
        "LOSS_INACTIVE_TAIL_SLOTS": "2024325",
        "TRAIN_SAMPLES": "19709952",
        "TRAIN_ITERS": "19248",
        "LR_WARMUP_ITERS": "400",
        "LR_WSD_DECAY_SAMPLES": "3942400",
        "CURRICULUM_ORDER_MODE": "randomized",
        "DATA_SEED": "20260609",
        "MIX_SEED": "20260801",
        "ROTARY_BASE": "500000",
        "ROPE_SCALING_FACTOR": "8.0",
    }
    require(env == expected_env, f"cpt.env drift: {env!r}")
    trainer = (ROOT / "scripts/train/train_apertus_cpt.sbatch").read_text(encoding="utf-8")
    require("--no-check-for-nan-in-loss-and-grad" not in trainer, "generic trainer disables NaN/Inf checks")

    print(json.dumps({
        "ok": True,
        "recipe_id": recipe["recipe_id"],
        "active_tokens": tokens["total"],
        "updates": batch["training_updates"],
        "vocab_size": tokenizer["total_vocab_size"],
        "launch_ready": False,
        "pending_launch_gates": recipe["launch_gates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
