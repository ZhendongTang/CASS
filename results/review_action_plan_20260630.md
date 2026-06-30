# Review action plan, 2026-06-30

Source review: `审稿意见/审稿意见1.docx`

## Changes already applied to the draft

- Retitled the paper from a "multitask ICU LLM" framing to "Multiple ICU Prediction Tasks" to avoid implying joint multitask training.
- Rewrote the CASS description to clarify that the actual model input is the deterministic `ClinTime-FeatureSummary/1` pipe-delimited concept-summary record.
- Replaced the tunable salience-score equation with the implementation-consistent fixed priority tuple: intervention, abnormality, count, latest time, concept tie-break.
- Added a short-context permutation diagnostics table to make the row-order sensitivity motivation visible in the main results section.
- Updated the code-availability URL to `https://github.com/ZhendongTang/CASS`.
- Regenerated and synchronized the key result figures and result-summary tables into the Overleaf clean draft directory.

## Remaining high-priority experiments

- CASS hash-equality audit under legal same-time reorderings across all centers.
- CASS prediction-equality audit using deterministic inference on reordered records.
- Larger or full-test-set evaluation with bootstrap 95% confidence intervals.
- Structured-feature baselines using the same CASS aggregate features, ideally LightGBM/XGBoost.
- External-center sanity checks for below-random AUROC, especially score inversion and label mapping.
- CASS ablations: remove trend, abnormality, provenance, salience, and compare hourly-block CASS against full-window CASS.

## Framing to preserve

- Supported: compactness, source-center performance, reuse of the same input format across endpoints.
- Partially supported: external mortality transfer.
- Not supported: serialization alone solves multicenter domain shift.
