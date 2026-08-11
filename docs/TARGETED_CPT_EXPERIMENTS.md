# Targeted 8B follow-up experiments

Two follow-ups are being prepared after the complete stationary 8B CPT run.
Their portable machine contract is
[`targeted_8b_followups.json`](../configs/training/targeted_8b_followups.json).
The receipt-producing implementation lives in
`fffoivos/train-apertus-with-glossapi/subprojects/08_targeted_8b_cpt_experiments`.

## A: academic, HPLT and polytonic mixture

The modern stream contains one pass over `openarchives.gr`, one pass over
`greek_phd`, the same post-decontamination number of active HPLT tokens, and one
pass over the existing 14,929-document `poly_train` split. HPLT first uses a
deterministic SHA-256 identity quarter as a capacity pool; after exclusions and
packing, the frozen seeded catalog prefix is cut to exactly the academic active
token total. These sources are
randomized into one stationary stream; there is no academic-first curriculum.
Foreign replay remains 20% and the inherited Greek source-family replay remains
1% at every point.

The exact `poly_train` file must be resolved from an already-existing
repository, Hugging Face, or CSCS artifact and audited against the frozen split
manifest before A can freeze. It must not be reconstructed or replaced by a
different Ancient-Greek corpus; there is no remote-home dependency.

The source is the public Apertus-standard anonymized dataset revision
`987b8955fcd395c6219e39df9e64715457f69065`. The selected rows are scanned
against the pinned GreekMMLU revision after anonymization. This scan is an
explicit decontamination exclusion with a decision ledger. No second global
deduplication is allowed.

Before packing, every selected document is also compared by exact UTF-8 text
hash against all 13 frozen validation panels. Exact matches are excluded with
a separate ledger and a zero-overlap post-scan. This preserves the old panels
as genuine heldouts; it is not deduplication, and non-validation duplicates
retain their original multiplicity.

Planning arithmetic, pending the exact polytonic count and new contamination
removals, is 25.548B active tokens and 6,092 updates. WSD-10 occupies the final
20%; AdEMAMix alpha/beta3 ramps are scaled to the exact new horizon.

## B: continue the best checkpoint

This arm resumes the exact optimizer/model/RNG state at parent update 9,536.
It consumes every non-HPLT packed sequence in the unconsumed suffix of the
parent D0 schedule and adds only replay sequences that also occur after that
prefix. “Unseen” therefore means unseen packed token spans; a long document may
have had an earlier span consumed.

The remaining non-HPLT mass is 9,123,187,023 active tokens. WSD-10 starts
immediately and decays over the full continuation. Optimizer warmup and
AdEMAMix ramps are not restarted or rescaled. Planning geometry is 2,754
updates, ending at absolute update 12,290.

## Evaluation and resources

Both arms keep the 13 content-clean source-conditioned panels and report
GreekMMLU accuracy, choice NLL and correct-answer BPB. A evaluates GreekMMLU
about every 2B tokens; B about every 1B.

The initial 13-panel checkpoint evaluation runs in four sequential groups on
one four-GPU `debug` node. It invokes the already proven per-group scorer and
publishes the output tree only after all 13 receipts pass. This changes queue
overhead, not panel identity, model geometry, tokenizer or metric arithmetic.

Metadata, decontamination, packing control, receipts, lightweight smokes,
conversion and evaluation control run on one-node Clariden `debug`
allocations. One prelaunch test cannot fit there: after all debug-built assets
are frozen, each arm uses one bounded 16-node, one-leaf `normal` allocation to
compare two uninterrupted DP32 updates with one update, a synchronous
checkpoint, and one resumed update. The three trajectories share that
allocation and execute four optimizer updates total. Logged loss and parameter
norm must match exactly; the predeclared inherited DP32 gradient-norm bound is
`atol=0.001`, `rtol=0.02`. Production remains blocked until this receipt passes.

The nested scheduler proof is rebound to every executing immutable bundle. Its
controller and child both run on `debug`; the controller runs inside the
production uenv and submits its child with `--uenv-passthrough=ignore`, while
the child verifies the same bundle and rank-local torchrun/Megatron runtime.
This scheduler-control proof never consumes a `normal` allocation.

Production training itself uses only the proven 16-node DP32 profile. DP64
remains prohibited because it failed trajectory parity despite its speedup.

For A, only one successor may be pending. With a conservative 10-hour segment,
20-minute reserve and 12-hour allocation, the maximum harmless hold is 100
minutes and the request trigger is 500 minutes after the source segment starts.
A fresh live capacity and leaf-switch snapshot is mandatory before each normal
submission.
