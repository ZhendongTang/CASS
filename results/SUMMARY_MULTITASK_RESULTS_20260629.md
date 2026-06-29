# Summary-format multitask results

Date: 2026-06-29

Remote source: `/home/tang/pces_icu/work/tasks/llm_summary_20260629/`

Local copy: `refine-logs/remote-results/llm_summary_20260629/`

Setup: Llama-3-8B, Summary format, max_length 2048, seed 42, MIMIC training, three-center evaluation with 500 records per center.

## Results

| Run | Task | Train records | Eval center | AUROC | AUPRC | Brier | ECE |
|---|---|---:|---|---:|---:|---:|---:|
| T501 | LOS >48h | 5k | MIMIC | 0.5571 | 0.5408 | 0.2553 | 0.0721 |
| T501 | LOS >48h | 5k | eICU | 0.5579 | 0.4376 | 0.2539 | 0.1271 |
| T501 | LOS >48h | 5k | NWICU | 0.5029 | 0.4796 | 0.2963 | 0.1869 |
| T502 | LOS >48h | 10k | MIMIC | 0.6193 | 0.5862 | 0.2459 | 0.0722 |
| T502 | LOS >48h | 10k | eICU | 0.5756 | 0.4661 | 0.2594 | 0.1485 |
| T502 | LOS >48h | 10k | NWICU | 0.5114 | 0.5123 | 0.3135 | 0.2472 |
| T503 | Future intervention | 5k | MIMIC | 0.7742 | 0.4209 | 0.0572 | 0.0538 |
| T503 | Future intervention | 5k | eICU | 0.5793 | 0.0688 | 0.0573 | 0.0492 |
| T503 | Future intervention | 5k | NWICU | 0.4925 | 0.1224 | 0.1203 | 0.1162 |
| T504 | Future intervention | 10k | MIMIC | 0.8000 | 0.5013 | 0.0566 | 0.0551 |
| T504 | Future intervention | 10k | eICU | 0.5221 | 0.0636 | 0.0588 | 0.0569 |
| T504 | Future intervention | 10k | NWICU | 0.5173 | 0.1482 | 0.1334 | 0.1335 |

## Interpretation

1. Summary generalizes beyond mortality on the source center. The future-intervention task is strong on MIMIC and improves from 0.7742 to 0.8000 AUROC when scaling from 5k to 10k records.
2. LOS >48h shows weaker but nonzero signal. Scaling from 5k to 10k improves MIMIC AUROC from 0.5571 to 0.6193 and eICU from 0.5579 to 0.5756.
3. External-center transfer is mixed. eICU retains modest signal on LOS and future intervention, while NWICU is near random for LOS and weak for future intervention.
4. These results support a cautious multitask claim: Summary is a useful LLM event narration format across multiple ICU endpoints on the source center, but cross-center generalization remains task- and center-dependent.

## Paper wording

Supported: "Summary-format event narration improves LLM learnability beyond mortality, with source-center gains on LOS and future-intervention prediction and partial external transfer."

Not supported: "Summary uniformly generalizes across all centers and tasks."
