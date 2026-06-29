# Summary mortality 20k seed-stability results

Date: 2026-06-29

Setup: Llama-3-8B, ClinTime-Summary, MIMIC-IV training only, 20k records, max_length 2048, auto-pos-weight, one epoch, 500 test records per center.

## AUROC/AUPRC/Brier/ECE

| Run | Seed | Center | AUROC | AUPRC | Brier | ECE |
|---|---:|---|---:|---:|---:|---:|
| P302 | 42 | MIMIC | 0.7780 | n/a | n/a | n/a |
| P302 | 42 | eICU | 0.4673 | n/a | n/a | n/a |
| P302 | 42 | NWICU | 0.6086 | n/a | n/a | n/a |
| S601 | 43 | MIMIC | 0.7875 | 0.3666 | 0.1229 | 0.1418 |
| S601 | 43 | eICU | 0.5897 | 0.1450 | 0.0848 | 0.0564 |
| S601 | 43 | NWICU | 0.6585 | 0.2116 | 0.1342 | 0.1545 |
| S602 | 44 | MIMIC | 0.7614 | 0.3336 | 0.1035 | 0.0871 |
| S602 | 44 | eICU | 0.5832 | 0.1193 | 0.0852 | 0.0605 |
| S602 | 44 | NWICU | 0.6085 | 0.2007 | 0.1504 | 0.1557 |

## Interpretation

The two new 20k seeds confirm that the mortality result is not a seed42 spike. MIMIC-IV AUROC remains in the 0.7614--0.7875 range across seed42/43/44, with seed43/44 also improving eICU AUROC over the earlier seed42 run. NWICU remains consistently above 0.60 AUROC. The paper can therefore use Summary 20k as the main mortality result, while still stating that external transfer is center-dependent.
