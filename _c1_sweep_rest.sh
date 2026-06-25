#!/bin/bash
# Salvage finish: b05 already lives in c1v4_b05 (copied from the mis-namespaced
# `main` run). This trains+evals only the two lost cells (b01, b03) with the
# FIXED naming (EXP= via make, not AO_EXP env), then builds the sweep dashboard.
# Mining is cached, so training is near-instant; the eval suite is the real cost.
#   cd /workspace/ao && nohup ./_c1_sweep_rest.sh > _c1_sweep_rest.log 2>&1 &
set -uo pipefail
cd /workspace/ao
PY=/workspace/ao/envs/train/bin/python

step () { echo "===== [$(date +%H:%M:%S)] STEP: $* ====="; }
fail () { echo "!!! FAILED at: $* — aborting"; exit 1; }

for b in 0.1 0.3; do
  tag=c1v4_b$(echo "$b" | tr -d '.')
  step "TRAIN $tag (beta=$b, epochs=1)"
  AO_DPO_BETA="$b" AO_DPO_EPOCHS=1 make rl EXP="$tag"   || fail "train $tag"
  step "EVAL $tag"
  make eval EXP="$tag"                                  || fail "eval $tag"
done

step "sweep dashboard (baseline + b01 + b03 + b05 + c1v3)"
$PY -m ao_cli.sweep_dashboard \
  --baseline baseline_replication \
  --runs c1v3 c1v4_b01 c1v4_b03 c1v4_b05 \
  || echo "!! sweep dashboard failed (non-fatal)"

echo "===== [$(date +%H:%M:%S)] REST-OF-SWEEP COMPLETE ====="
echo "Compare betas: artifacts/Qwen3-8B/aobench_results/b_sweep_dashboard.html"
