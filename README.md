# Causal Dialog Route Reranker

This is SE team's compliance-first alternative entry for the DialNav Challenge.
It is a single online Navigator-Guide rollout built on the official RAINbow
code and frozen checkpoints.

The agent boundary is explicit:

1. Only the Navigator owns and invokes the question-generation model.
2. The Navigator sends a natural-language question to the Guide.
3. The Guide localizes that question and generates a natural-language route
   answer. The Guide has no question-generation model.
4. Only the question and answer strings enter the Navigator's instruction
   state. The Guide keeps its route-generation state internally. An explicit
   stop is encoded by the Guide as answer text and interpreted from that text
   by the Navigator.

The method does not use question fingerprints, target descriptions, shared QG,
language DFS, visual signatures, multiple rollouts, episode selection, hosted
APIs, or evaluation labels.

## Frozen entry

- Team: `SE team`
- Method: `Causal Dialog Route Reranker`
- Upstream: RAINbow commit `4ef165fad77e675026958e16d9e516e523192a33`
- Seed: `0`
- Submission: `submission.json`
- Submission SHA256:
  `52c191922ff14423efe5b290c0a9e1aa0fa0e094c06227145db3a421fbbf5efc`
- Counts: 91 val-seen, 241 val-unseen, 285 released Test
- Released-Test SR: 27.3684% (78/285); hidden Test Score unknown

See `COMPLIANCE_RESPONSE.md`, `TECHNICAL_REPORT_ONE_PAGE.pdf`, and
`REPRODUCIBILITY.md` before running the code.

Public repository:
<https://github.com/neilyysun-ui/dialnav-causal-dialog-route-reranker>
