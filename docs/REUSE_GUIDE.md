# Reuse Guide

This repository separates training-regime choices (preserve) from operational
launch profiles (tune per run). This guide draws that boundary.

## Preserve Unless You Are Testing A Change

These files encode the project choices and should be reused as-is for a Greek
Apertus CPT run unless the experiment explicitly changes that choice:

- [`configs/tokenizer/extension.json`](../configs/tokenizer/extension.json)
- [`configs/tokenizer/removal_policy.json`](../configs/tokenizer/removal_policy.json)
- [`configs/training/cpt.env`](../configs/training/cpt.env)
- [`configs/training/cpt.regime.json`](../configs/training/cpt.regime.json)
- [`docs/HYPERPARAMETERS.md`](HYPERPARAMETERS.md)

This includes tokenizer size/alignment, model geometry, optimizer, scheduler
policies, Goldfish loss, data semantics, checkpoint semantics, and tokenizer
extension construction.

Two point exceptions live inside these files, flagged where they appear:

- the data seed — run-relative; see "Tune Per Run" below.
- the three sweep-candidate values β2 / α / peak LR — validated starting points,
  not frozen results (`open_decisions.sweep_candidates` in the manifest, `†` in
  the hyperparameter doc).

## Tune Per Run

These choices depend on the cluster, allocation, queue state, target wall time,
reproducibility, and debugging needs:

- node count
- walltime
- partition/account
- software image
- transport profile
- segment length
- segment count
- data seed (`DATA_SEED`) — the default is a run-date placeholder, not a tuned
  value; pick a fresh seed per run
- whether to run a smoke test before the full launch

Use [`configs/training/launch_profiles/`](../configs/training/launch_profiles)
as examples. A profile records a setup that worked; its resource choices are not
training-regime invariants.

## Useful Reusable Scripts

- [`scripts/train/submit_cpt_chain.sh`](../scripts/train/submit_cpt_chain.sh)
  preserves checkpoint-boundary segmented launch logic.
- [`scripts/train/train_apertus_cpt.sbatch`](../scripts/train/train_apertus_cpt.sbatch)
  preserves the Megatron command construction and runtime guards.
- [`scripts/train/patch_megatron_gpt_dataset_no_shuffle.py`](../scripts/train/patch_megatron_gpt_dataset_no_shuffle.py)
  applies the physical-order curriculum patch.
- [`scripts/train/verify_megatron_curriculum_indices.py`](../scripts/train/verify_megatron_curriculum_indices.py)
  verifies the generated GPTDataset cache order.

Before reusing a script, check whether it consumes the training regime, a launch
profile, or local path inputs. Do not turn a launch-profile value into a
training-regime invariant without an explicit experiment rationale.
