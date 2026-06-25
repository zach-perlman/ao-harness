#!/bin/bash
# C1 DPO beta sweep, fully self-contained: baseline eval + 3 train/eval cells.
#   - pair count comes from config.yaml (now n_tokens=1800, pairs_per_token=4)
#   - beta varies per cell via AO_DPO_BETA (b01 = "more pairs, same beta as c1v3",
#     isolating the data-scale effect; b03/b05 test tighter KL anchoring)
#   - the mined DPO dataset is identical across betas, so the FIRST cell mines and
#     caches it (artifacts/<model>/dpo_cache/); b03/b05 reload it and skip mining.
#     (Force a fresh mine with AO_DPO_NOCACHE=1.)
#   - SWEEP SPEED: cells run AO_DPO_EPOCHS=1 (a full pass over all pairs is plenty to
#     RANK betas; ~2x faster than the default 2). train_dpo now micro-batches each
#     step's pairs into ONE forward, so a step is several-x faster too. Retrain the
#     WINNING beta at the full epochs=2 (drop AO_DPO_EPOCHS) for the final model.
# Each `make eval` runs the FULL suite and rebuilds dashboard.html vs baseline_replication.
# Run detached so you can leave:
#     cd /workspace/ao && nohup ./_c1_sweep.sh > _c1_sweep.log 2>&1 &
# then check progress with:  grep -aE "STEP|faithful|abstention_f1|done in|COMPLETE|FAILED" _c1_sweep.log
set -uo pipefail
cd /workspace/ao
PY=/workspace/ao/envs/train/bin/python   # bare `python` isn't on PATH in this shell

step () { echo "===== [$(date +%H:%M:%S)] STEP: $* ====="; }
fail () { echo "!!! FAILED at: $* — aborting"; exit 1; }

# 1) refresh the baseline on the current (expanded) eval sets + deterministic judge
# step "baseline eval"
# make eval-baseline || fail "baseline eval"

# 2) train + full eval for each beta cell.
#   NOTE: the run name MUST go through make's EXP= variable, not an AO_EXP env var:
#   the Makefile sets `AO_EXP=$(EXP)` inline on every command, which would override
#   (clobber to "main") any AO_EXP we exported here. AO_DPO_* are read directly by
#   rl.py from the env, so those stay as env vars.
for b in 0.1 0.3 0.5; do
  tag=c1v4_b$(echo "$b" | tr -d '.')
  step "TRAIN $tag (beta=$b, epochs=1)"
  AO_DPO_BETA="$b" AO_DPO_EPOCHS=1 make rl EXP="$tag"   || fail "train $tag"
  step "EVAL $tag"
  make eval EXP="$tag"                                  || fail "eval $tag"
done

# 3) custom side-by-side dashboard: baseline + all beta cells (+ c1v3 reference)
step "sweep dashboard"
$PY -m ao_cli.sweep_dashboard \
  --baseline baseline_replication \
  --runs c1v3 c1v4_b01 c1v4_b03 c1v4_b05 \
  || echo "!! sweep dashboard failed (non-fatal) — build manually with the same command"

echo "===== [$(date +%H:%M:%S)] SWEEP COMPLETE ====="
echo "Compare betas: open artifacts/Qwen3-8B/aobench_results/b_sweep_dashboard.html"
echo "Per-run detail: AO_EXP=c1v4_b01 python -m ao_cli dashboard   (repeat for b03, b05)"
