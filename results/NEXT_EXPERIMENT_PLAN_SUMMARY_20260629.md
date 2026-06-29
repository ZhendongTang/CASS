# Next Experiment Plan: Summary-format ICU Event Narration

Date: 2026-06-29

## Problem

The paper should now center on Summary as an effective ICU event narration format for LLMs. Existing results support strong source-center performance on mortality and future-intervention prediction, but the evidence is still incomplete for cross-center generalization, seed stability at the best scale, and whether Summary itself is the causal representation improvement.

## Method Thesis

ClinTime-Summary compresses high-frequency ICU event streams into a compact clinical narration that is easier for LLMs to learn from than markup-heavy or row-like serializations, while retaining enough temporal and clinical signal to generalize across multiple ICU endpoints.

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1: Summary is the best current LLM input family for ICU event prediction | This is the paper's core contribution | Summary beats XML/compact/JSON/flat under matched backbone, train size, context length, and task | B1, B2 |
| C2: Summary learns task-relevant clinical signal beyond mortality | Prevents the method from looking task-specific | Mortality, LOS, and future-intervention results show source-center signal and some external transfer | B3 |
| C3: Remaining failures are domain shift and calibration, not just format failure | Explains eICU/NWICU limits honestly | Mixed-center training or center calibration improves external AUROC/ECE without destroying MIMIC | B4 |
| Anti-claim: gains only come from more tokens or more training data | Reviewer concern | Matched-token/matched-scale comparisons and deletion studies preserve Summary advantage | B2, B5 |

## Current Evidence

### Completed summary evidence

| Setting | Task | Train | MIMIC AUROC | eICU AUROC | NWICU AUROC |
|---|---|---:|---:|---:|---:|
| Summary 2048, seed42 | mortality | 5k | 0.6950 | 0.4753 | 0.6091 |
| Summary 2048, seed43 | mortality | 5k | 0.7188 | 0.5453 | 0.5712 |
| Summary 2048, seed44 | mortality | 5k | 0.7089 | 0.4959 | 0.5857 |
| Summary 2048, seed42 | mortality | 10k | 0.7034 | 0.5174 | 0.6377 |
| Summary 2048, seed42 | mortality | 20k | 0.7780 | 0.4673 | 0.6086 |
| Summary 2048, seed42 | LOS >48h | 5k | 0.5571 | 0.5579 | 0.5029 |
| Summary 2048, seed42 | LOS >48h | 10k | 0.6193 | 0.5756 | 0.5114 |
| Summary 2048, seed42 | future intervention | 5k | 0.7742 | 0.5793 | 0.4925 |
| Summary 2048, seed42 | future intervention | 10k | 0.8000 | 0.5221 | 0.5173 |

### Evidence gap

1. 20k mortality has only seed42. It is the strongest number, so it needs seed43/44.
2. Auxiliary tasks have only seed42 and no matched structured baselines at 5k/10k.
3. Summary lacks direct matched 2048 comparisons against compact/XML/flat on the same 5k/10k settings.
4. Cross-center failures are not diagnosed. eICU and NWICU need either mixed-center training or post-hoc calibration analysis.

## Experiment Blocks

### B1: Stabilize the best Summary mortality result

- Claim tested: C1.
- Why this exists: 20k Summary seed42 reaches 0.7780 MIMIC AUROC, which is paper-defining but not yet seed-stable.
- Dataset / task: mortality, MIMIC train, MIMIC/eICU/NWICU eval, 500 eval records per center for direct comparison with prior LLM runs.
- Compared systems: Summary 20k 2048 auto-pos-weight, seeds 42/43/44.
- Metrics: AUROC primary; AUPRC, Brier, ECE, sensitivity@90 specificity secondary.
- Runs:
  - `S601`: mortality, Summary 20k 2048 auto-pos-weight seed43.
  - `S602`: mortality, Summary 20k 2048 auto-pos-weight seed44.
- Success criterion: MIMIC mean >=0.74 and no seed collapses below 0.70.
- Failure interpretation: seed42 was optimistic; paper should report 5k 3-seed as primary and 20k as exploratory scale signal.
- Table / figure target: Main Table 1 and scale curve.
- Priority: MUST-RUN.

### B2: Matched representation comparison

- Claim tested: C1 and anti-claim.
- Why this exists: Reviewers will ask whether Summary wins because it is a better representation or because previous XML/compact runs used weaker settings.
- Dataset / task: mortality, MIMIC 5k train, max_length 2048, seed42, same eval records.
- Compared systems:
  - Summary 2048 existing `P203`.
  - Compact 2048.
  - XML 2048.
  - Flat 2048.
  - Optional JSON 2048 only if cheap.
- Metrics: AUROC/AUPRC/Brier/ECE plus token length and truncation/coverage.
- Runs:
  - `S611`: Compact 5k 2048 seed42 auto-pos-weight.
  - `S612`: XML 5k 2048 seed42 auto-pos-weight.
  - `S613`: Flat 5k 2048 seed42 auto-pos-weight.
  - `S614`: token audit for Summary/XML/Compact/Flat/JSON on the exact 500 eval records.
- Success criterion: Summary improves MIMIC AUROC by >=0.05 over every alternative and has lower or comparable token count than XML.
- Failure interpretation: Summary advantage is not format-specific; the contribution should shift to scaling/context rather than narration design.
- Table / figure target: Main Table 2, token-efficiency figure.
- Priority: MUST-RUN.

### B3: Multitask seed replication

- Claim tested: C2.
- Why this exists: T501-T504 show useful signal but only seed42. The paper can claim multitask generality only if at least one auxiliary task is seed-stable.
- Dataset / tasks: LOS >48h and future intervention, MIMIC train, three-center eval.
- Compared systems: Summary 10k 2048 auto-pos-weight, seeds 42/43/44.
- Metrics: AUROC primary; AUPRC especially for future intervention because prevalence is low; calibration secondary.
- Runs:
  - `S621`: LOS >48h, Summary 10k 2048 seed43.
  - `S622`: LOS >48h, Summary 10k 2048 seed44.
  - `S623`: Future intervention, Summary 10k 2048 seed43.
  - `S624`: Future intervention, Summary 10k 2048 seed44.
- Success criterion: MIMIC source-center AUROC remains >0.58 for LOS and >0.75 for future intervention in mean across seeds.
- Failure interpretation: auxiliary-task evidence is single-seed exploratory; keep it in appendix and avoid multitask claims in the abstract.
- Table / figure target: Main or appendix multitask table depending on stability.
- Priority: MUST-RUN for future intervention; SHOULD-RUN for LOS if compute is tight.

### B4: Cross-center adaptation and calibration

- Claim tested: C3.
- Why this exists: Summary is strong on MIMIC but external transfer is mixed. A paper needs a diagnosis and possibly a simple fix.
- Dataset / tasks: mortality first; auxiliary tasks only if mortality adaptation works.
- Compared systems:
  - MIMIC-only Summary 10k/20k.
  - MIMIC+eICU mixed training.
  - MIMIC+NWICU mixed training.
  - MIMIC+eICU+NWICU mixed training.
  - Post-hoc temperature or Platt calibration per center.
- Metrics: external AUROC, AUPRC, Brier, ECE; center-wise macro AUROC for eICU.
- Runs:
  - `S631`: mortality, Summary MIMIC 10k + eICU 10k, 2048 seed42.
  - `S632`: mortality, Summary MIMIC 10k + NWICU all/10k, 2048 seed42.
  - `S633`: mortality, Summary three-center mixed 20k total, 2048 seed42.
  - `S634`: post-hoc calibration on existing P203/P301/P302 predictions if prediction files are available.
- Success criterion: eICU AUROC >=0.58 or ECE decreases by >=0.03 without MIMIC AUROC dropping below 0.70.
- Failure interpretation: external domain shift is structural; paper should report this as a limitation and avoid broad deployment claims.
- Table / figure target: External generalization table and calibration plot.
- Priority: MUST-RUN for S631 and S634; NICE-TO-HAVE for S632/S633.

### B5: Summary deletion study

- Claim tested: anti-claim and simplicity.
- Why this exists: The paper should show which pieces of Summary matter and avoid a vague "natural language is better" claim.
- Dataset / task: mortality, MIMIC 5k, max_length 2048, seed42.
- Compared systems:
  - Full Summary existing P203.
  - No trend language.
  - No abnormal/flag language.
  - Last-value-only summary.
  - Counts-only summary.
- Metrics: AUROC/AUPRC/Brier/ECE; token count.
- Runs:
  - `S641`: Summary-no-trend 5k 2048 seed42.
  - `S642`: Summary-no-abnormal-flag 5k 2048 seed42.
  - `S643`: Summary-last-value-only 5k 2048 seed42.
  - `S644`: Summary-counts-only 5k 2048 seed42.
- Success criterion: Full Summary is best or near-best, and at least one clinically meaningful component deletion causes a measurable AUROC drop.
- Failure interpretation: current Summary may be over-specified; simplify the representation and improve paper elegance.
- Table / figure target: Ablation table.
- Priority: SHOULD-RUN.

### B6: Matched structured baselines

- Claim tested: C1 and frontier necessity.
- Why this exists: The paper should not compare Summary LLM to full-data HistGBDT only. It needs matched-sample baselines.
- Dataset / tasks: mortality, LOS, future intervention; MIMIC train sizes 5k/10k/20k where applicable.
- Compared systems: HistGBDT and Event Transformer on exact LLM sample IDs.
- Metrics: AUROC/AUPRC/Brier/ECE.
- Runs:
  - `S651`: HistGBDT matched mortality 5k/10k/20k.
  - `S652`: Event Transformer matched mortality 5k/20k.
  - `S653`: HistGBDT matched LOS 5k/10k and future-intervention 5k/10k.
  - `S654`: Event Transformer matched LOS/future-intervention 10k if runtime is acceptable.
- Success criterion: Summary narrows the gap to structured baselines, especially on future intervention; if it does not beat them, the paper frames Summary as an LLM interface rather than a new SOTA predictor.
- Failure interpretation: LLM remains less sample-efficient; report honestly.
- Table / figure target: Main baseline table.
- Priority: MUST-RUN for S651/S653; SHOULD-RUN for S652/S654.

## Run Order and Gates

| Priority | Runs | Goal | Gate | Estimated cost |
|---:|---|---|---|---:|
| 1 | S601-S602 | Verify 20k mortality seed stability | If mean MIMIC <0.74, stop claiming 20k as main result | 5-7 GPUh |
| 2 | S611-S614 | Prove Summary beats matched formats | If Summary does not win, reframe contribution before more scale runs | 4-8 GPUh |
| 3 | S623-S624 | Stabilize future-intervention multitask claim | If MIMIC mean <0.75, move multitask to appendix | 3-5 GPUh |
| 4 | S631 + S634 | Diagnose eICU/domain shift | If no improvement, write limitation and stop adaptation search | 2-4 GPUh |
| 5 | S651/S653 | Produce fair structured baseline table | Required for paper credibility | CPU/GPU mixed |
| 6 | S621/S622, S641-S644, S632/S633 | Fill appendix and mechanism | Run only after main tables are secure | variable |

## Immediate Launch Queue

Launch these first:

1. `S601`: mortality Summary 20k 2048 auto-pos-weight seed43.
2. `S602`: mortality Summary 20k 2048 auto-pos-weight seed44.
3. `S611`: mortality Compact 5k 2048 auto-pos-weight seed42.
4. `S612`: mortality XML 5k 2048 auto-pos-weight seed42.
5. `S613`: mortality Flat 5k 2048 auto-pos-weight seed42.

Do not launch deletion studies until these finish. The representation comparison may change which ablations are worth running.

## Paper-ready Decision Rules

- If S601/S602 confirm 20k MIMIC AUROC >=0.74, use Summary 20k as main mortality result.
- If S601/S602 are unstable, use 5k three-seed mortality as main result and report 20k as scale exploration.
- If S611-S613 show Summary wins by >=0.05 AUROC, the title and abstract can center on Summary event narration.
- If future-intervention seeds remain strong, keep "multitask" in the main contribution.
- If future-intervention seeds are unstable, write "additional endpoint evidence" instead of "multitask generalization."
- If mixed-center training does not fix eICU/NWICU, explicitly state that Summary improves LLM learnability but does not solve center shift.

## Final Checklist

- [ ] 20k mortality has 3 seeds.
- [ ] Summary has matched 2048 comparisons against XML/compact/flat.
- [ ] Future-intervention multitask claim has at least 3 seeds or is moved to appendix.
- [ ] Structured baselines are matched by train size and task.
- [ ] Cross-center failures are diagnosed rather than hidden.
- [ ] Summary deletion study either supports the design or simplifies it.
