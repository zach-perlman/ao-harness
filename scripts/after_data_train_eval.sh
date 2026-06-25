#!/bin/bash
# =============================================================================
# after_data_train_eval.sh — chain baseline train+eval onto a running `make data`
#
# PURPOSE
#   `make data` (corpus -> convqa -> evalsets) is already running in another
#   terminal. This watcher waits for it to exit, confirms it actually produced
#   a COMPLETE dataset, and then runs the clean baseline `make train` followed
#   by `make eval` — unattended, surviving shell/agent exit.
#
# MECHANISM
#   1. Poll until the given PID disappears (kill -0), re-checking that the PID
#      is still the original `make data` process so a reused PID can't fool us.
#   2. Gate on the three terminal artifacts of a full data run; if any is
#      missing we assume data failed/was interrupted and do NOT burn GPU on a
#      half-baked corpus. (make stops on first error, so a partial run is the
#      failure mode we guard against.)
#   3. Run `make train` (EXP empty => the no-contribution baseline AO), PROMOTE
#      its checkpoint to the frozen reference (checkpoints/replication_v1, where
#      baseline.ao_lora points), then `make eval-baseline` (self-manages its own
#      judge) to populate aobench_results/baseline_replication — the curve every
#      future contribution is compared against. Timestamped; a failure at any
#      step aborts the rest.
#
#   Promotion layout: `evaluate --baseline` uses baseline.ao_lora verbatim as the
#   adapter path, so replication_v1 must itself BE the adapter dir — we copy the
#   CONTENTS of the trained run's final/ into it (not final/ as a subdir).
# =============================================================================
set -u

PID="${1:?usage: after_data_train_eval.sh <make-data-pid>}"
REPO=/workspace/ao
SLUG=gemma-4-E4B-it
ART="$REPO/artifacts/$SLUG"
LOG="$ART/auto_train_eval.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# --- 1. wait for the make-data process to finish ----------------------------
log "watching make data (pid $PID); will train+eval baseline on success"
while kill -0 "$PID" 2>/dev/null; do
  # bail if the PID got recycled into something that is no longer our make run
  if ! grep -qa make "/proc/$PID/cmdline" 2>/dev/null; then
    log "pid $PID is no longer a make process; aborting watcher"
    exit 1
  fi
  sleep 30
done
log "make data (pid $PID) exited"

# --- 2. confirm the data run actually completed -----------------------------
corpus="$ART/corpus/corpus.jsonl"
convqa="$ART/convqa/train.parquet"
evalsets="$ART/aobench_datasets"
missing=0
[ -s "$corpus" ]                                   || { log "MISSING corpus: $corpus"; missing=1; }
[ -s "$convqa" ]                                   || { log "MISSING convqa: $convqa"; missing=1; }
[ -d "$evalsets" ] && [ -n "$(ls -A "$evalsets" 2>/dev/null)" ] || { log "MISSING evalsets: $evalsets"; missing=1; }
if [ "$missing" -ne 0 ]; then
  log "data run incomplete -> NOT training. Re-run \`make data\`, then retrigger."
  exit 1
fi
log "data artifacts present -> proceeding to baseline train+eval"

# --- 3. baseline train -> promote -> eval-baseline --------------------------
cd "$REPO" || { log "cannot cd $REPO"; exit 1; }

log "START make train (baseline, EXP empty)"
if ! make train >>"$LOG" 2>&1; then
  log "make train FAILED -> skipping promote+eval (see log above)"
  exit 1
fi
log "DONE make train"

# Promote the freshly trained run's adapter to the frozen baseline reference.
main_final="$ART/checkpoints/ao_${SLUG}_v2/final"
baseline_ckpt="$ART/checkpoints/replication_v1"
if [ ! -f "$main_final/adapter_config.json" ]; then
  log "MISSING trained adapter at $main_final -> cannot promote"
  exit 1
fi
rm -rf "$baseline_ckpt" && mkdir -p "$baseline_ckpt" && cp -a "$main_final/." "$baseline_ckpt/"
log "PROMOTED $main_final -> $baseline_ckpt (this run is now the baseline)"

log "START make eval-baseline (-> aobench_results/baseline_replication)"
if ! make eval-baseline >>"$LOG" 2>&1; then
  log "make eval-baseline FAILED (see log above)"
  exit 1
fi
log "DONE make eval-baseline — baseline established + evaluated"
