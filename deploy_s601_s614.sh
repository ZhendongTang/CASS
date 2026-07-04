#!/usr/bin/env bash
# Deploy S601-S614 experiment queue to GPU server
# Usage: bash deploy_s601_s614.sh
set -euo pipefail

SERVER="root@192.168.10.6"
REMOTE_DIR="/home/tang/pces_icu"
LOCAL_MANIFEST="experiment_queue/2026-06-29T0853Z/manifest.json"
TAG="summary_next"

echo "=== Syncing code (audit_token_lengths.py) ==="
scp scripts/audit_token_lengths.py ${SERVER}:${REMOTE_DIR}/scripts/audit_token_lengths.py

echo "=== Syncing queue manager ==="
scp tools/queue_manager_tmux.py ${SERVER}:${REMOTE_DIR}/tools/queue_manager_tmux.py

echo "=== Uploading manifest ==="
scp ${LOCAL_MANIFEST} ${SERVER}:${REMOTE_DIR}/manifests/manifest_${TAG}.json

echo "=== Launching queue manager on server ==="
ssh ${SERVER} << 'ENDSSH'
cd /home/tang/pces_icu

# Kill any existing queue manager for this tag
tmux kill-session -t "EQ_summary_next" 2>/dev/null || true

# Create log dir
mkdir -p work/final_v1/summary_next/logs
mkdir -p manifests

# Launch in tmux
tmux new-session -d -s "EQ_summary_next" \
  "nohup python3 tools/queue_manager_tmux.py \
    --manifest manifests/manifest_summary_next.json \
    --state work/final_v1/summary_next/queue_state.json \
    --log-dir work/final_v1/summary_next/logs \
    > work/final_v1/summary_next/queue_mgr.log 2>&1"

echo "Queue manager launched in tmux session: EQ_summary_next"
echo "Monitor with: tmux attach -t EQ_summary_next"
echo "Log: tail -f work/final_v1/summary_next/queue_mgr.log"
ENDSSH

echo "=== Done! Queue: S601 -> S602 -> S611 -> S612 -> S613 -> S614 ==="
