# Model and data card

- Method: `Causal Dialog Route Reranker`
- Base code: official RAINbow commit
  `4ef165fad77e675026958e16d9e516e523192a33`
- Frozen models: DST Navigator, LANA question generator, LANA answer generator,
  and GTL localizer
- New or fine-tuned parameters: none
- Selection data: 806 DialNav training episodes from 30 scan-disjoint scans
- External data inherited through official RAINbow: RAIN/RxR/R2R/CVDN and
  Matterport3D assets under their original terms
- Additional external datasets: none
- Hosted APIs: none
- Seed: 0
- Hardware used for final inference: one NVIDIA H100 GPU
- Maximum actions: 50
- Batch size: 8
- WTA: confidence 0.7 with a two-step cooldown
- Temporal reranker: GTL top-5, geodesic weight 0.5
- Answer path limit: 20
- Released-Test execution: one saved frozen run on 285 episodes
- Known limitation: unseen localization remains weak and Guide confirmations
  can be wrong when localization is wrong
