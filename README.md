# CASS ICU

CASS is a code release for Compact Attribute-aware Serialization of ICU event streams for prediction with large language models. The repository contains cohort construction, event extraction, serialization, training, evaluation, audit, and figure/table builder code. It intentionally excludes raw ICU data, model weights, checkpoints, LoRA adapters, predictions, experiment results, and rendered figures.

## Model Weights

Download model weights from the official provider pages and keep them outside the git repository:

| Backbone | Official weights |
| --- | --- |
| Llama-3-8B | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
| GLM-4-9B | [zai-org/glm-4-9b-chat](https://huggingface.co/zai-org/glm-4-9b-chat) |
| Gemma-2-9B | [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b) |

The experiment manifests assume local paths such as `/data/models/Meta-Llama-3-8B`, `/data/models/glm-4-9b-chat`, and `/data/models/gemma-2-9b`. Adjust those paths in the manifests if your server layout differs.

## Datasets

This repository does not redistribute clinical data. Request access from the official dataset pages and follow each license/data use agreement:

| Dataset | Official source | Expected local root example |
| --- | --- | --- |
| MIMIC-IV v3.1 | [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/) | `/data/icu-data/mimiciv/3.1` |
| eICU Collaborative Research Database v2.0 | [PhysioNet eICU-CRD](https://physionet.org/content/eicu-crd/2.0/) | `/data/icu-data/eicu-crd/2.0` |
| Northwestern ICU database v0.1.0 | [PhysioNet NWICU](https://physionet.org/content/nwicu-northwestern-icu/) | `/data/icu-data/nwicu-northwestern-icu/0.1.0/data` |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[llm,test]'
```

If CUDA-enabled PyTorch is already installed on the server, keep that build and install the package without replacing PyTorch:

```bash
pip install -e . --no-deps
pip install pandas scikit-learn transformers accelerate peft bitsandbytes pytest
```

## Experiment Server Requirements

The final experiments fine-tune 8B/9B class LLMs with QLoRA and then run multi-view audits. A single modern CUDA GPU is sufficient for the provided manifests, while multiple GPUs can be used by running independent manifest jobs.

Recommended server:

| Component | Requirement |
| --- | --- |
| GPU | NVIDIA GPU with at least 24 GB VRAM for one 8B/9B QLoRA job; 40 GB or more is recommended for longer contexts, larger batches, and faster audits. |
| CUDA | CUDA-compatible PyTorch 2.2 or newer; keep the server's tested CUDA/PyTorch build if already installed. |
| CPU/RAM | 16 or more CPU cores and at least 128 GB system RAM for large table scans and concurrent data preprocessing. |
| Disk | At least 1 TB free local SSD/NVMe space for credentialed ICU datasets, derived event JSONL files, checkpoints, logs, and audit outputs. |
| OS | Linux server environment with Bash. The queue manifests assume POSIX shell commands and CUDA-visible GPUs. |
| Python | Python 3.10 or newer. |

Operational notes:

- Keep raw data under a read-only location such as `/data/icu-data`.
- Keep model weights under a non-git location such as `/data/models`.
- Keep all generated files under `work/`; this directory is ignored by git.
- For the queue manager, use one GPU per running job unless you have explicitly adjusted `gpus` and `max_parallel` in the manifest.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `TRANSFORMERS_NO_TF=1`, and `TOKENIZERS_PARALLELISM=false` as in the generated manifests.

## Data Processing Pipeline

The pipeline writes all derived artifacts under `work/`, which is ignored by git.

```text
official raw data
  -> leakage-safe 6h landmark cohorts
  -> canonical event JSONL files
  -> schema/window audit
  -> optional derived tasks
  -> LLM/baseline training manifests
  -> evaluation and figure/table builder scripts
```

Build adult first-ICU-stay cohorts:

```bash
pces-icu build-cohort \
  --dataset mimic \
  --data-root /data/icu-data/mimiciv/3.1 \
  --output work/cohorts/mimic.csv

pces-icu build-cohort \
  --dataset eicu \
  --data-root /data/icu-data/eicu-crd/2.0 \
  --output work/cohorts/eicu.csv

pces-icu build-cohort \
  --dataset nwicu \
  --data-root /data/icu-data/nwicu-northwestern-icu/0.1.0/data \
  --output work/cohorts/nwicu.csv
```

Extract first-window canonical ICU events:

```bash
pces-icu build-events \
  --dataset mimic \
  --data-root /data/icu-data/mimiciv/3.1 \
  --cohort work/cohorts/mimic.csv \
  --output work/events/mimic.jsonl.gz

pces-icu build-events \
  --dataset eicu \
  --data-root /data/icu-data/eicu-crd/2.0 \
  --cohort work/cohorts/eicu.csv \
  --output work/events/eicu.jsonl.gz

pces-icu build-events \
  --dataset nwicu \
  --data-root /data/icu-data/nwicu-northwestern-icu/0.1.0/data \
  --cohort work/cohorts/nwicu.csv \
  --output work/events/nwicu.jsonl.gz
```

Audit extracted event streams:

```bash
pces-icu audit-events --data work/events/mimic.jsonl.gz --output work/events/mimic.audit.json
pces-icu audit-events --data work/events/eicu.jsonl.gz --output work/events/eicu.audit.json
pces-icu audit-events --data work/events/nwicu.jsonl.gz --output work/events/nwicu.audit.json
```

Build the four public serialization views from the same canonical event files:

```bash
pces-icu build-serializations \
  --events work/events/mimic.jsonl.gz \
  --output-root work/serializations/mimic

pces-icu build-serializations \
  --events work/events/eicu.jsonl.gz \
  --output-root work/serializations/eicu

pces-icu build-serializations \
  --events work/events/nwicu.jsonl.gz \
  --output-root work/serializations/nwicu
```

Each command writes `cass.jsonl.gz`, `json.jsonl.gz`, `xml.jsonl.gz`, `meds.jsonl.gz`, and `manifest.json` under the selected output directory. The CASS view uses the paper's compact feature-summary serializer, JSON/XML use canonical non-permuted ClinTime views, and MEDS uses the row-wise long-table baseline. To materialize a subset, repeat `--format`, for example `--format cass --format json`.

Optional derived tasks use the same event schema:

```bash
pces-icu build-los-task \
  --events work/events/mimic.jsonl.gz \
  --cohort work/cohorts/mimic.csv \
  --output work/tasks/los_gt48/mimic.jsonl.gz

pces-icu build-future-intervention-task \
  --events work/events/mimic_24h.jsonl.gz \
  --output work/tasks/future_intervention/mimic.jsonl.gz
```

## Training And Audits

Run the final 20k multi-backbone experiment queue by generating a manifest and executing it with the queue runner:

```bash
python scripts/build_final_manifest.py \
  --output work/manifests/final_20k_manifest.json \
  --root work/final_v1_20k

python tools/run_manifest_sequential.py \
  --manifest work/manifests/final_20k_manifest.json \
  --state work/final_v1_20k/queue_state.json \
  --log-dir work/final_v1_20k/logs
```

You can also run one LLM training job directly:

```bash
python -m pces_icu.multicenter_llm train \
  --data work/events/mimic.jsonl.gz \
  --model /data/models/Meta-Llama-3-8B \
  --output work/runs/cass_llama_seed42 \
  --variant summary \
  --seed 42 \
  --sample-seed 2026 \
  --epochs 1 \
  --max-length 2048 \
  --grad-accum 8 \
  --lr 2e-5 \
  --kl-weight 0.2 \
  --max-train-records 20000 \
  --max-val-records 20000 \
  --eval-data mimic=work/events/mimic.jsonl.gz \
  --eval-data eicu=work/events/eicu.jsonl.gz \
  --eval-data nwicu=work/events/nwicu.jsonl.gz \
  --max-eval-records 20000
```

## Figure And Table Builders

The repository includes scripts, not generated results. All commands below write to `work/` so generated CSV/Markdown/TeX/SVG artifacts stay outside git.

### Table 1: Dataset Summary

Experiment input: the leakage-safe cohort CSV files and canonical 6-hour event JSONL files for MIMIC-IV, eICU, and NWICU.

Statistical computation: `scripts/build_table1_dataset_summary.py` counts patients, stays, hospitals, records, total events, median events per stay, positive label rate, and split sizes directly from the derived cohort/event files.

Table build command:

```bash
python scripts/build_table1_dataset_summary.py \
  --dataset mimic=work/events/mimic.jsonl.gz \
  --dataset eicu=work/events/eicu.jsonl.gz \
  --dataset nwicu=work/events/nwicu.jsonl.gz \
  --cohort mimic=work/cohorts/mimic.csv \
  --cohort eicu=work/cohorts/eicu.csv \
  --cohort nwicu=work/cohorts/nwicu.csv \
  --csv work/tables/table1_dataset_summary.csv \
  --markdown work/tables/table1_dataset_summary.md \
  --tex work/tables/table1_dataset_summary.tex
```

### Table 2: Representation Stability

Experiment input: permutation/canonicalization audit metrics from representation stress runs. These are produced by the Table 2 audit manifest and audit scripts, for example:

```bash
python scripts/build_table2_permutation_formats_manifest.py \
  --output work/manifests/table2_format_permutation.json \
  --root work/final_v2/table2_format_permutation

python tools/run_manifest_sequential.py \
  --manifest work/manifests/table2_format_permutation.json \
  --state work/final_v2/table2_format_permutation/queue_state.json \
  --log-dir work/final_v2/table2_format_permutation/logs
```

Statistical computation: `scripts/build_table2_representation_summary.py` aggregates legal surface-perturbation metrics across completed seed/backbone audit files. It reports mean prediction standard deviation, maximum probability span, and label flip rate as mean plus standard deviation across source metrics files.

Table build command:

```bash
python scripts/build_table2_representation_summary.py \
  --root work/final_v2/table2_format_permutation \
  --center mimic \
  --csv work/tables/table2_representation_summary.csv \
  --markdown work/tables/table2_representation_summary.md \
  --tex work/tables/table2_representation_summary.tex
```

### Table 3: Endpoint Prediction Performance

Experiment input: final 20k multi-backbone training/evaluation outputs under `work/final_v1_20k/main`, generated by `scripts/build_final_manifest.py` and `tools/run_manifest_sequential.py`.

Statistical computation: `scripts/build_table3_endpoint_summary.py` reads each `metrics.json` from `main/<backbone>/<variant>_seed*/evaluation/<center>/`, groups by backbone, serialization variant, and center, and reports AUROC, AUPRC, Brier score, and ECE as mean plus standard deviation across completed seeds.

Table build command:

```bash
python scripts/build_table3_endpoint_summary.py \
  --root work/final_v1_20k \
  --csv work/tables/table3_endpoint_summary.csv \
  --markdown work/tables/table3_endpoint_summary.md \
  --tex work/tables/table3_endpoint_summary.tex
```

For paired patient-level comparisons and bootstrap confidence intervals, run the companion analysis script:

```bash
python scripts/analyze_complete_results.py \
  --root work/final_v1_20k \
  --output-dir work/analysis/final_v1_20k \
  --bootstrap-reps 2000 \
  --bootstrap-seed 2026
```

### Fig. 2: Format Token Efficiency

Experiment input: token-budget audits from `work/final_v1_20k/token_audit` and endpoint metrics from `work/final_v1_20k/main`.

Statistical computation: `scripts/draw_fig2_format_token_efficiency.py` averages token length, truncation rate, evidence-retention rate, predictive metric, and metric-per-1000-token efficiency over the supplied audit/evaluation files.

Plot build command:

```bash
python scripts/draw_fig2_format_token_efficiency.py \
  --token-root work/final_v1_20k/token_audit \
  --metrics-root work/final_v1_20k/main \
  --csv work/figures/fig2_format_token_efficiency.csv \
  --svg work/figures/fig2_format_token_efficiency.svg
```

## Sanity Checks

```bash
python scripts/sanity.py
python -m unittest discover -s tests -v
```

No API keys, raw clinical data, model checkpoints, predictions, rendered figures, or result summaries should be committed.
