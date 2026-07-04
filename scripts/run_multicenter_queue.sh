#!/usr/bin/env bash
set -uo pipefail

export PYTHONPATH=/home/tang/pces_icu/src
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false

cd /home/tang/pces_icu || exit 1
source .venv/bin/activate

ROOT=work/multicenter_v2/pilot_1000
mkdir -p "$ROOT/logs"
printf '{"state":"running","started_at":"%s"}\n' "$(date --iso-8601=seconds)" > "$ROOT/queue_status.json"

run_one() {
  local variant="$1"
  local seed="$2"
  local output="$ROOT/${variant}_seed${seed}"
  local log="$ROOT/logs/${variant}_seed${seed}.log"
  if [[ -s "$output/train_metrics.json" && -s "$output/evaluation/mimic/metrics.json" && -s "$output/evaluation/eicu/metrics.json" && -s "$output/evaluation/nwicu/metrics.json" ]]; then
    printf 'SKIP completed %s seed=%s\n' "$variant" "$seed" | tee -a "$ROOT/queue.log"
    return 0
  fi
  printf 'START %s seed=%s %s\n' "$variant" "$seed" "$(date --iso-8601=seconds)" | tee -a "$ROOT/queue.log"
  python -m pces_icu.multicenter_llm train \
    --data work/events/mimic.jsonl.gz \
    --model /data/models/glm-4-9b-chat \
    --output "$output" \
    --variant "$variant" \
    --seed "$seed" \
    --epochs 1 \
    --max-length 256 \
    --grad-accum 8 \
    --lr 2e-5 \
    --kl-weight 0.2 \
    --max-train-records 1000 \
    --max-val-records 256 \
    --eval-data mimic=work/events/mimic.jsonl.gz \
    --eval-data eicu=work/events/eicu.jsonl.gz \
    --eval-data nwicu=work/events/nwicu.jsonl.gz \
    --max-eval-records 500 > "$log" 2>&1
  local status=$?
  printf 'END %s seed=%s status=%s %s\n' "$variant" "$seed" "$status" "$(date --iso-8601=seconds)" | tee -a "$ROOT/queue.log"
  return "$status"
}

failures=0
for seed in 42 43 44; do
  for variant in flat xml xml_kl; do
    run_one "$variant" "$seed" || failures=$((failures + 1))
  done
done

if [[ "$failures" -eq 0 ]]; then
  state=complete
else
  state=completed_with_failures
fi
printf '{"state":"%s","failures":%d,"finished_at":"%s"}\n' "$state" "$failures" "$(date --iso-8601=seconds)" > "$ROOT/queue_status.json"
