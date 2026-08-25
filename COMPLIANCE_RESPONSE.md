# Response to the shared-QG concern

**Team:** SE team  
**Alternative method:** Causal Dialog Route Reranker

We agree that using one question-generation module at both the target and the
Navigator's current location can bypass the intended communication problem.
The alternative entry does not use that mechanism.

## Agent ownership and communication

- The **Navigator alone** owns the frozen LANA question generator and invokes
  it on its current observation.
- The **Guide has no question generator**. It owns the frozen GTL localizer and
  LANA answer generator, receives the Navigator's question text, estimates the
  referenced location, and returns a natural-language route answer.
- The Navigator receives only the natural-language question and answer strings.
  It never receives a target node, target image, target description, graph
  route, localization logits, or Guide feature vector.
- When the Guide estimates that the Navigator is at a goal, it returns the
  fixed natural-language sentence "You are already at the target location.
  Stop here now; do not move anywhere else." The Navigator's controller decides
  to stop by matching this received answer text; it receives no confirmation
  bit, target-match index, or other side channel.
- The Guide may retain the route produced by its own previous answer. This is
  private Guide-side temporal state and is used only to rerank the Guide's next
  localization candidates. It is not sent to the Navigator as structured data.

The relevant construction is visible in
`source_snapshot/external/RAINbow/holistic/main.py:set_agents`: the QG object is
passed only to `ModularNavigator`; `ModularGuide` receives only the answer and
localization models. The runtime dialog call is `navigator.ask(...)`, followed
by `guide.localize(...)`, `guide.answer(...)`, and finally
`navigator.update_instruction(...)` with text.

## Explicit exclusions

This entry uses one causal rollout per episode. It has no shared QG, question
fingerprint, target-language signature, target-description transfer, language
DFS, visual ambiguity guard, cached evaluation lookup, trajectory selector,
path splicing, manual evaluation annotation, Gemini, UniAPI, or other hosted
inference API.

The exact submission has 617 records and SHA256
`52c191922ff14423efe5b290c0a9e1aa0fa0e094c06227145db3a421fbbf5efc`.
Its released-Test SR is 27.3684% (78/285). This is not claimed as the hidden
ranking Test Score.
