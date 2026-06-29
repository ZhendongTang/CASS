# Experiment Tracker: Summary Next Runs

Date: 2026-06-29

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| S601 | B1 | 20k mortality seed stability | Llama Summary, 20k, 2048, auto-pos-weight, seed43 | MIMIC train -> MIMIC/eICU/NWICU | AUROC/AUPRC/Brier/ECE | MUST | RUNNING | Launched in remote screen `SUMMARY_NEXT_Q1`; output `work/summary_next_20260629/s601_summary_20k_l2048_autopw_s43` |
| S602 | B1 | 20k mortality seed stability | Llama Summary, 20k, 2048, auto-pos-weight, seed44 | MIMIC train -> MIMIC/eICU/NWICU | AUROC/AUPRC/Brier/ECE | MUST | TODO | Main mortality result needs 3 seeds |
| S611 | B2 | matched representation comparison | Llama Compact, 5k, 2048, auto-pos-weight, seed42 | MIMIC train -> MIMIC/eICU/NWICU | AUROC/AUPRC/Brier/ECE/token | MUST | TODO | Compare to existing Summary P203 |
| S612 | B2 | matched representation comparison | Llama XML, 5k, 2048, auto-pos-weight, seed42 | MIMIC train -> MIMIC/eICU/NWICU | AUROC/AUPRC/Brier/ECE/token | MUST | TODO | Compare to existing Summary P203 |
| S613 | B2 | matched representation comparison | Llama Flat, 5k, 2048, auto-pos-weight, seed42 | MIMIC train -> MIMIC/eICU/NWICU | AUROC/AUPRC/Brier/ECE/token | MUST | TODO | Determines whether Summary beats plain sequence at same context |
| S614 | B2 | token/coverage audit | Summary/XML/Compact/Flat/JSON | same eval records | token count/truncation/coverage | MUST | TODO | Required for "compact" claim |
| S621 | B3 | LOS seed replication | Llama Summary, LOS, 10k, 2048, seed43 | MIMIC train -> MIMIC/eICU/NWICU | AUROC/AUPRC/Brier/ECE | SHOULD | TODO | Run if LOS remains in main paper |
| S622 | B3 | LOS seed replication | Llama Summary, LOS, 10k, 2048, seed44 | MIMIC train -> MIMIC/eICU/NWICU | AUROC/AUPRC/Brier/ECE | SHOULD | TODO | Run if LOS remains in main paper |
| S623 | B3 | future-intervention seed replication | Llama Summary, future intervention, 10k, 2048, seed43 | MIMIC train -> MIMIC/eICU/NWICU | AUROC/AUPRC/Brier/ECE | MUST | QUEUED | Queued in remote screen `SUMMARY_AUX_FUTURE_SEEDS`; waits for `SUMMARY_NEXT_Q1` to finish |
| S624 | B3 | future-intervention seed replication | Llama Summary, future intervention, 10k, 2048, seed44 | MIMIC train -> MIMIC/eICU/NWICU | AUROC/AUPRC/Brier/ECE | MUST | QUEUED | Queued in remote screen `SUMMARY_AUX_FUTURE_SEEDS`; runs after S623 |
| S631 | B4 | eICU adaptation | Llama Summary, MIMIC 10k + eICU 10k, 2048, seed42 | mixed train -> MIMIC/eICU/NWICU | AUROC/AUPRC/Brier/ECE | MUST | TODO | Tests whether eICU failure is train-domain shift |
| S632 | B4 | NWICU adaptation | Llama Summary, MIMIC 10k + NWICU 10k/all, 2048, seed42 | mixed train -> MIMIC/eICU/NWICU | AUROC/AUPRC/Brier/ECE | NICE | TODO | Run only if S631 is informative |
| S633 | B4 | three-center training | Llama Summary, mixed three-center 20k total, 2048, seed42 | mixed train -> MIMIC/eICU/NWICU | AUROC/AUPRC/Brier/ECE | NICE | TODO | Tests broad adaptation |
| S634 | B4 | post-hoc calibration | Existing Summary mortality predictions | center-specific validation/test | Brier/ECE/AUROC | MUST | TODO | Cheap if prediction files exist |
| S641 | B5 | summary deletion | Summary-no-trend, mortality, 5k, 2048, seed42 | MIMIC train -> 3 centers | AUROC/AUPRC/Brier/ECE/token | SHOULD | TODO | Mechanism ablation |
| S642 | B5 | summary deletion | Summary-no-abnormal-flag, mortality, 5k, 2048, seed42 | MIMIC train -> 3 centers | AUROC/AUPRC/Brier/ECE/token | SHOULD | TODO | Mechanism ablation |
| S643 | B5 | summary deletion | Summary-last-value-only, mortality, 5k, 2048, seed42 | MIMIC train -> 3 centers | AUROC/AUPRC/Brier/ECE/token | SHOULD | TODO | Simplicity ablation |
| S644 | B5 | summary deletion | Summary-counts-only, mortality, 5k, 2048, seed42 | MIMIC train -> 3 centers | AUROC/AUPRC/Brier/ECE/token | SHOULD | TODO | Minimal information control |
| S651 | B6 | matched structured baseline | HistGBDT mortality, 5k/10k/20k exact samples | MIMIC train -> 3 centers | AUROC/AUPRC/Brier/ECE | MUST | TODO | Fair baseline table |
| S652 | B6 | matched structured baseline | Event Transformer mortality, 5k/20k exact samples | MIMIC train -> 3 centers | AUROC/AUPRC/Brier/ECE | SHOULD | TODO | Strong sequence baseline |
| S653 | B6 | matched structured baseline | HistGBDT LOS/future intervention, 5k/10k exact samples | MIMIC train -> 3 centers | AUROC/AUPRC/Brier/ECE | MUST | TODO | Fair multitask baseline |
| S654 | B6 | matched structured baseline | Event Transformer LOS/future intervention, 10k exact samples | MIMIC train -> 3 centers | AUROC/AUPRC/Brier/ECE | SHOULD | TODO | Only if runtime acceptable |

## First Queue

1. S601
2. S602
3. S611
4. S612
5. S613
6. S614

Decision after first queue: if Summary does not clearly beat matched Compact/XML/Flat, stop the Summary-centered paper framing and rewrite the claim around context/scale instead.
