#!/bin/bash
cd /workspace/eval
for RATE in 0.8 0.9; do
  NAME="uniattn${RATE}_l03"
  echo "===== $NAME RESTART $(date +%H:%M:%S) ====="
  python3 restart_enc_uniattn.py $RATE attention_uniattn 0.3 || { echo "restart FAILED $NAME"; continue; }
  echo "===== $NAME EVAL $(date +%H:%M:%S) ====="
  bash run_config.sh "$NAME"
  echo "===== $NAME DONE $(date +%H:%M:%S) ====="
done
echo "QUALITY_SWEEP_DONE $(date +%H:%M:%S)"
