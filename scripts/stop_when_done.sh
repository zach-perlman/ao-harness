#!/bin/bash
# Watch the running `make data` job and stop this Vast instance when it ends, so
# GPU charges halt the moment the pipeline finishes (storage is still billed; the
# container + disk survive a `stop`, so you can restart and resume).
#
# Mechanism: poll `kill -0 <PID>` until the target process is gone (exit reason
# doesn't matter — success or crash both free the GPU), then call the in-container
# vastai CLI (authenticated by the Vast-set CONTAINER_API_KEY) to stop CONTAINER_ID.
set -u

PID="${1:?usage: stop_when_done.sh <pid>}"
LOG=/workspace/ao/scripts/stop_when_done.log

echo "[$(date -u +%FT%TZ)] watching pid $PID (CONTAINER_ID=$CONTAINER_ID); will stop instance on exit" >>"$LOG"

while kill -0 "$PID" 2>/dev/null; do
    sleep 30
done

echo "[$(date -u +%FT%TZ)] pid $PID ended — stopping instance $CONTAINER_ID" >>"$LOG"

for attempt in 1 2 3; do
    if vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >>"$LOG" 2>&1; then
        echo "[$(date -u +%FT%TZ)] stop request accepted (attempt $attempt)" >>"$LOG"
        exit 0
    fi
    echo "[$(date -u +%FT%TZ)] stop attempt $attempt failed; retrying in 30s" >>"$LOG"
    sleep 30
done
echo "[$(date -u +%FT%TZ)] ERROR: all stop attempts failed — instance still running" >>"$LOG"
exit 1
