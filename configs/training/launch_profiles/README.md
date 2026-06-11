# Launch Profiles

Launch profiles are operational examples, not training-regime invariants.

They capture hard-won cluster setup but deliberately leave per-run resource
choices — node count, walltime, account, partition, software image — for you to
reconsider every time. The training regime itself lives in
[`../cpt.env`](../cpt.env) and [`../cpt.regime.json`](../cpt.regime.json).

Source a profile through:

```bash
LAUNCH_PROFILE_ENV=configs/training/launch_profiles/clariden_16node_cxi.env \
  scripts/train/submit_cpt_chain.sh
```
