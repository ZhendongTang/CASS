#!/usr/bin/env bash
set -euo pipefail

cd /home/tang/pces_icu
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export CLINTIME_NO_4BIT=1
export PYTHONPATH=src

PY=/root/anaconda3/bin/python
MODEL=/data/models/Meta-Llama-3-8B
ROOT=work/tasks/llm_improve_s42
LOS=work/tasks/los_gt48/mimic.jsonl.gz
FUTURE=work/tasks/future_intervention/mimic.jsonl.gz
EICU_FUTURE=work/tasks/future_intervention/eicu.jsonl.gz
NWICU_FUTURE=work/tasks/future_intervention/nwicu.jsonl.gz
mkdir -p "$ROOT"

LOS_PROMPT="Task: Predict whether ICU length of stay will exceed 48 hours using only events from the first 6 ICU hours. Return a risk score for LOS >48h."
FUTURE_PROMPT="Task: Predict whether a new target intervention starts after ICU hour 6 and before hour 24. Target interventions are vasopressors, invasive/noninvasive ventilation, intubation, or renal replacement therapy. Patients with target intervention before hour 6 were excluded. Use only first-6h events. Return a risk score."

run_one() {
  local name="$1"
  shift
  echo "===== ${name} $(date -Is) ====="
  "$PY" -m pces_icu.multicenter_llm train "$@" 2>&1 | tee "$ROOT/${name}.log"
  echo "===== done ${name} $(date -Is) ====="
}

run_one los_compact_overfit128_e10 \
  --data "$LOS" --model "$MODEL" --output "$ROOT/los_compact_overfit128_e10" \
  --variant compact_kl --seed 42 --sample-seed 2026 --epochs 10 --max-length 1024 \
  --grad-accum 8 --lr 2e-4 --max-train-records 128 --max-val-records 128 \
  --train-eval-records 128 --max-eval-records 256 --task-prompt "$LOS_PROMPT"

run_one los_summary_overfit128_e10 \
  --data "$LOS" --model "$MODEL" --output "$ROOT/los_summary_overfit128_e10" \
  --variant summary --seed 42 --sample-seed 2026 --epochs 10 --max-length 512 \
  --grad-accum 8 --lr 2e-4 --max-train-records 128 --max-val-records 128 \
  --train-eval-records 128 --max-eval-records 256 --task-prompt "$LOS_PROMPT"

run_one los_summary_posw_e3 \
  --data "$LOS" --model "$MODEL" --output "$ROOT/los_summary_posw_e3" \
  --variant summary --seed 42 --sample-seed 2026 --epochs 3 --max-length 512 \
  --grad-accum 8 --lr 2e-4 --max-train-records 1000 --max-val-records 256 \
  --train-eval-records 256 --max-eval-records 500 --task-prompt "$LOS_PROMPT" \
  --eval-data mimic="$LOS"

run_one future_summary_posw_e3 \
  --data "$FUTURE" --model "$MODEL" --output "$ROOT/future_summary_posw_e3" \
  --variant summary --seed 42 --sample-seed 2026 --epochs 3 --max-length 512 \
  --grad-accum 8 --lr 2e-4 --auto-pos-weight --max-train-records 1000 --max-val-records 256 \
  --train-eval-records 256 --max-eval-records 500 --task-prompt "$FUTURE_PROMPT" \
  --eval-data mimic="$FUTURE" --eval-data eicu="$EICU_FUTURE" --eval-data nwicu="$NWICU_FUTURE"
