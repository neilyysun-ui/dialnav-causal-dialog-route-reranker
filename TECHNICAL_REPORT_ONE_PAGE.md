# Causal Dialog Route Reranker

**Team:** SE team  
**Alternative DialNav Challenge entry, August 2026**

## Method and communication boundary

We submit a single-rollout dialog-navigation method built on the official
RAINbow system. Only the Navigator owns and invokes the frozen LANA question
generator. At a dialog step, the Navigator describes its current observation
as a natural-language question. The Guide, which has no QG module, localizes
that question with frozen GTL and generates a natural-language route answer
with the frozen LANA answer generator. Only the question and answer strings
enter the Navigator's instruction state.

Our addition is Guide-private temporal localization. If the previous Guide
answer was generated from route R at step t0, the expected location at step t
is the route node indexed by min(max(t-t0,0), |R|-1). For GTL top-five candidate
v with probability p(v), the Guide selects the minimum of
`-log(max(p(v),1e-12)) + 0.5*dG(v,e_t)`. Without prior route state, it uses GTL
top-1. The state never crosses to the Navigator as structured data. A Guide
goal confirmation is also natural-language text. The system executes one
causal trajectory, with no replay or episode selector.

This alternative uses no shared QG, QFP, target-language signature,
target-description transfer, language DFS, visual signature, trajectory
splicing, evaluation cache/label, manual evaluation annotation, Gemini,
UniAPI, or hosted inference API.

## Data, models, selection, and execution

The base is RAINbow commit `4ef165fad77e675026958e16d9e516e523192a33`.
Frozen components are the official DST Navigator, LANA QG, LANA answer
generator, and GTL localizer; no weights were trained or fine-tuned. Their
upstream data provenance is RAIN/RxR/R2R/CVDN and Matterport3D. No additional
dataset was used.

Top-k 5 and weight 0.5 were fixed using 806 train episodes from 30
scan-disjoint scans. Relative to the same-fold base, train Score increased from
60.2713 to 64.2928 (+4.0215 points), with scan-clustered 95% CI
[+2.1385,+6.0464]. Final inference used seed 0, one NVIDIA H100 GPU, Python
3.10, PyTorch 2.10, batch size 8, WTA `ctc_0.7_2`, answer-path limit 20, and a
50-action cap.

## Results and limitations

| Split | Episodes | Score | SR | Mean steps | Mean dialogs |
|---|---:|---:|---:|---:|---:|
| val-seen | 91 | 0.5977 | 65.93% | 21.59 | 3.26 |
| val-unseen | 241 | 0.2646 | 31.95% | 24.59 | 5.29 |
| released Test | 285 | hidden | 27.37% | 27.37 | 6.05 |

The released-Test result is 78/285 and is not the hidden ranking Test Score.
The method was frozen from train-only evidence before its saved Test run; the
official outcomes did not alter its parameters. Its main limitation is weak
unseen localization: an incorrect Guide localization can still produce an
incorrect route or confirmation.

## Reproducibility

The delivery includes the source overlay, fixed configuration, tests, exact
submission, checksums, model/data card, and runtime-weight hashes. Submission
SHA256: `52c191922ff14423efe5b290c0a9e1aa0fa0e094c06227145db3a421fbbf5efc`.
