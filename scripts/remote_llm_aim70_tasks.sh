#!/usr/bin/env bash
set -euo pipefail

cd /home/tang/pces_icu
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export CLINTIME_NO_4BIT=1
export PYTHONPATH=src

PY=/root/anaconda3/bin/python
MODEL=/data/models/Meta-Llama-3-8B
ROOT=work/tasks/llm_aim70_s42
FUTURE=work/tasks/future_intervention/mimic.jsonl.gz
EICU_FUTURE=work/tasks/future_intervention/eicu.jsonl.gz
NWICU_FUTURE=work/tasks/future_intervention/nwicu.jsonl.gz
LOS=work/tasks/los_gt48/mimic.jsonl.gz
mkdir -p "$ROOT"

FUTURE_PROMPT="Task: Predict whether a new target intervention starts after ICU hour 6 and before hour 24. Target interventions are vasopressors, invasive/noninvasive ventilation, intubation, or renal replacement therapy. Patients with target intervention before hour 6 were excluded. Use only first-6h events. Return a risk score."
LOS_PROMPT="Task: Predict whether ICU length of stay will exceed 48 hours using only events from the first 6 ICU hours. Return a risk score for LOS >48h."

run_one() {
  local name="$1"
  shift
  if [[ -f "$ROOT/$name/train_metrics.json" ]]; then
    echo "===== skip ${name}: train_metrics.json exists ====="
    return 0
  fi
  echo "===== ${name} $(date -Is) ====="
  "$PY" -m pces_icu.multicenter_llm train "$@" 2>&1 | tee "$ROOT/${name}.log"
  echo "===== done ${name} $(date -Is) ====="
}

# First verify that balanced sampling makes the future task learnable.
run_one future_summary_bal_overfit256_e8 \
  --data "$FUTURE" --model "$MODEL" --output "$ROOT/future_summary_bal_overfit256_e8" \
  --variant summary --seed 42 --sample-seed 2026 --epochs 8 --max-length 512 \
  --grad-accum 8 --lr 1e-4 --balance-train --max-train-records 256 --max-val-records 256 \
  --train-eval-records 256 --max-eval-records 500 --task-prompt "$FUTURE_PROMPT"

# Main AUROC target runs: balanced positives/negatives, no extreme class weight.
run_one future_summary_bal4000_lr1e4_e3 \
  --data "$FUTURE" --model "$MODEL" --output "$ROOT/future_summary_bal4000_lr1e4_e3" \
  --variant summary --seed 42 --sample-seed 2026 --epochs 3 --max-length 512 \
  --grad-accum 8 --lr 1e-4 --balance-train --max-train-records 4000 --max-val-records 512 \
  --train-eval-records 512 --max-eval-records 1000 --task-prompt "$FUTURE_PROMPT" \
  --eval-data mimic="$FUTURE" --eval-data eicu="$EICU_FUTURE" --eval-data nwicu="$NWICU_FUTURE"

run_one future_summary_bal4000_lr5e5_e5 \
  --data "$FUTURE" --model "$MODEL" --output "$ROOT/future_summary_bal4000_lr5e5_e5" \
  --variant summary --seed 42 --sample-seed 2026 --epochs 5 --max-length 512 \
  --grad-accum 8 --lr 5e-5 --balance-train --max-train-records 4000 --max-val-records 512 \
  --train-eval-records 512 --max-eval-records 1000 --task-prompt "$FUTURE_PROMPT" \
  --eval-data mimic="$FUTURE" --eval-data eicu="$EICU_FUTURE" --eval-data nwicu="$NWICU_FUTURE"

# Check whether compact keeps event timing/detail that summary loses.
run_one future_compact_bal4000_lr1e4_e3 \
  --data "$FUTURE" --model "$MODEL" --output "$ROOT/future_compact_bal4000_lr1e4_e3" \
  --variant compact_kl --seed 42 --sample-seed 2026 --epochs 3 --max-length 1024 \
  --grad-accum 8 --lr 1e-4 --balance-train --max-train-records 4000 --max-val-records 512 \
  --train-eval-records 512 --max-eval-records 1000 --task-prompt "$FUTURE_PROMPT" \
  --eval-data mimic="$FUTURE" --eval-data eicu="$EICU_FUTURE" --eval-data nwicu="$NWICU_FUTURE"

# LOS control: if this remains random while future improves, the issue is task-dependent.
run_one los_summary_bal4000_lr1e4_e3 \
  --data "$LOS" --model "$MODEL" --output "$ROOT/los_summary_bal4000_lr1e4_e3" \
  --variant summary --seed 42 --sample-seed 2026 --epochs 3 --max-length 512 \
  --grad-accum 8 --lr 1e-4 --balance-train --max-train-records 4000 --max-val-records 512 \
  --train-eval-records 512 --max-eval-records 1000 --task-prompt "$LOS_PROMPT" \
  --eval-data mimic="$LOS"
