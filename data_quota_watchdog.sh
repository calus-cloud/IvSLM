#!/usr/bin/env bash
# Kills data download tmux sessions if fineweb100B/ exceeds the byte cap.
set -u
DIR="${IVLLM_DATA_DIR:-fineweb100B}"
CAP_BYTES="${IVLLM_DATA_CAP_BYTES:-$((50 * 1024 * 1024 * 1024))}"  # 50 GiB
INTERVAL="${IVLLM_DATA_WATCH_INTERVAL:-30}"
SESSIONS=(data_fine data_math data_code)

echo "[watchdog] dir=$DIR cap=$CAP_BYTES interval=${INTERVAL}s"
while true; do
    if [ -d "$DIR" ]; then
        SIZE=$(du -sb "$DIR" 2>/dev/null | awk '{print $1}')
        SIZE=${SIZE:-0}
        printf "[watchdog] %s  size=%s bytes  (%.2f GiB)\n" \
            "$(date -Iseconds)" "$SIZE" "$(echo "$SIZE/1073741824" | bc -l)"
        if [ "$SIZE" -ge "$CAP_BYTES" ]; then
            echo "[watchdog] CAP HIT (${SIZE} >= ${CAP_BYTES}). Killing sessions: ${SESSIONS[*]}"
            for s in "${SESSIONS[@]}"; do
                tmux kill-session -t "$s" 2>/dev/null && echo "[watchdog]   killed $s"
            done
            exit 0
        fi
    fi
    sleep "$INTERVAL"
done
