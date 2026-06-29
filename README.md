# ClinTime Summary ICU

This repository contains the public code and aggregate experiment records for the ClinTime Summary ICU experiments. The central claim is that a compact clinical-event narrative (`summary`) is an effective serialization format for ICU event streams: it is much smaller than XML-like representations, performs well on the main mortality task, and shows useful transfer across additional ICU prediction tasks.

## What Is Included

- `src/pces_icu/`: cohort construction, event extraction, serialization, metrics, classical baselines, and LLM fine-tuning/evaluation code.
- `configs/`: reproducible baseline and ablation configurations.
- `scripts/`: analysis, token-audit, result-manifest, and sanity-check utilities.
- `tests/`: unit tests for core data structures, metrics, serialization, and LLM helpers.
- `results/`: aggregate tables and experiment summaries used for paper drafting.

Raw ICU data, model checkpoints, LoRA adapters, prediction JSONL files, API response logs, and private server scripts are intentionally excluded.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[llm,test]'
```

On a server where CUDA-enabled PyTorch is already installed, keep that PyTorch build and install the package without dependencies:

```bash
pip install -e . --no-deps
pip install pandas scikit-learn transformers accelerate peft bitsandbytes pytest
```

## Data Layout

The code expects local, credential-free paths such as:

```text
work/
  cohorts/
  events/
  runs/
```

The released repository does not include MIMIC-IV, eICU, or NWICU records. Use your institutionally approved local copies and follow the dataset licenses.

## Build Event Streams

```bash
pces-icu build-cohort \
  --dataset mimic \
  --data-root /path/to/mimiciv/3.1 \
  --output work/cohorts/mimic.csv

pces-icu build-events \
  --dataset mimic \
  --data-root /path/to/mimiciv/3.1 \
  --cohort work/cohorts/mimic.csv \
  --output work/events/mimic.jsonl.gz

pces-icu audit-events \
  --data work/events/mimic.jsonl.gz \
  --output work/events/mimic.audit.json
```

Equivalent commands are available for `eicu` and `nwicu`.

## Train And Evaluate Summary LLM

```bash
python -m pces_icu.multicenter_llm train \
  --data work/events/mimic.jsonl.gz \
  --model /path/to/Meta-Llama-3-8B \
  --output work/runs/summary_20k_seed42 \
  --variant summary \
  --seed 42 \
  --sample-seed 2026 \
  --epochs 1 \
  --max-length 2048 \
  --grad-accum 8 \
  --lr 2e-5 \
  --kl-weight 0.2 \
  --auto-pos-weight \
  --max-train-records 20000 \
  --max-val-records 512 \
  --max-eval-records 500 \
  --train-eval-records 512 \
  --eval-data mimic=work/events/mimic.jsonl.gz \
  --eval-data eicu=work/events/eicu.jsonl.gz \
  --eval-data nwicu=work/events/nwicu.jsonl.gz
```

Auxiliary tasks such as length-of-stay and future-intervention prediction use the same command shape with task-specific event files and `--task-prompt`.

## Reported Results

The aggregate experiment summaries live in `results/`. Key files:

- `SUMMARY_MULTITASK_RESULTS_20260629.md`: current multitask evidence for summary serialization.
- `EXPERIMENT_TRACKER_SUMMARY_NEXT_20260629.md`: planned and queued robustness runs.
- `RESULTS.md`: earlier complete endpoint analysis.
- `paper_tables/*.tex`: LaTeX tables used in the draft paper.

As of the included results, the strongest completed evidence is:

- Summary 20k mortality seed43/44: MIMIC AUROC 0.7875/0.7614, eICU 0.5897/0.5832, NWICU 0.6585/0.6085.
- Future-intervention summary 10k seed42: MIMIC AUROC 0.8000, eICU 0.5221, NWICU 0.5173.
- Additional future-intervention seed43/44 runs were queued after the current main experiment queue.

## API Key Policy

DeepSeek-related utilities are included only for zero-shot audit reproducibility. No API key is stored in this repository. If those utilities are used, provide the key at runtime through the `DEEPSEEK_API_KEY` environment variable.

Do not commit `.env` files, API response logs, or raw prediction outputs.

## Sanity Checks

```bash
python scripts/sanity.py
python -m unittest discover -s tests -v
```
