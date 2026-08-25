# DialNav Challenge rules confirmation

**Leaderboard name:** SE team  
**Alternative method:** Causal Dialog Route Reranker

To the best of our knowledge, this alternative submission follows the DialNav
Challenge rules and the organizer's clarification concerning dialog-based
communication.

- It runs exactly one online trajectory per episode.
- Only the Navigator invokes question generation.
- The Guide does not contain or invoke a question-generation module.
- Navigator-to-Guide and Guide-to-Navigator messages are natural-language text.
- A Guide stop confirmation is transmitted as answer text and is interpreted
  by the Navigator from that text, without a separate confirmation signal.
- Guide target nodes and shortest paths remain Guide-private, as in the
  challenge protocol.
- No target node, image, feature, structured path, or localization output is
  communicated to the Navigator.
- No evaluation label, reference trajectory, reference dialog, future
  observation, or result cache is used at inference.
- No QFP, target-description sharing, language DFS, visual signature, multiple
  rollout selection, manual annotation, or hosted API is used.
- All external data, pretrained models, parameters, seeds, and execution
  details are disclosed in the technical report and model/data card.

The previous target-language-signature entry is withdrawn for prize
consideration. This alternative entry is the one we ask the organizers to
review.
