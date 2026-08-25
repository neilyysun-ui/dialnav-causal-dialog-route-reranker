# Execution provenance

The exact submitted JSON was assembled from the frozen val-seen and val-unseen
records and the saved single-run V6-A output on the organizer-released Test
input. The Test slice is canonically identical to that saved run and contains
285 unique instruction IDs.

- Full submission SHA256:
  `52c191922ff14423efe5b290c0a9e1aa0fa0e094c06227145db3a421fbbf5efc`
- Saved Test-run submission SHA256:
  `77c5f815ee392ec6321a09bc4836a20ee531880413111c8a04b270cd118c477f`
- Released-Test successes: 78/285
- Released-Test SR: 27.3684210526%
- Seed: 0
- Batch size: 8
- Maximum actions: 50

The public source is a compliance-only extraction of the active execution
path. Inactive research branches from the larger development workspace were
removed so that the submitted code contains no implementation of the rejected
shared-QG method. The temporal reranker, model calls, constants, checkpoint
interfaces, and single-rollout data flow used by this alternative are retained.
No official output was regenerated or modified during this packaging step.
