#!/bin/bash
cd /workspace/eval
for LAM in 0 0.3 0.7; do
  NAME="uniattn0.5_l${LAM/./}"
  echo "===== CONFIG $NAME RESTART $(date +%H:%M:%S) ====="
  python3 restart_enc_uniattn.py 0.5 attention_uniattn $LAM || { echo "restart FAILED $NAME"; continue; }
  echo "===== CONFIG $NAME EVAL $(date +%H:%M:%S) ====="
  bash run_config.sh "$NAME"
  echo "===== CONFIG $NAME DONE $(date +%H:%M:%S) ====="
done
echo "SWEEP_DONE $(date +%H:%M:%S)"
