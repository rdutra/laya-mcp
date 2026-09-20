# Coding-agent usage

`classify` is intended for small, explicit routing and relevance decisions—not for
passing entire files to a short-context model.

Good requests contain a compact fact summary and a concrete proposition or label set:

```json
{
  "context": "player_controller.gd: movement, jumping, acceleration, animation state",
  "question": "Is this file relevant to a player movement bug?",
  "decision_type": "noul"
}
```

For candidate routing, use `choice`:

```json
{
  "context": "Failure occurs only while landing after a jump.",
  "question": "Which subsystem is most likely relevant?",
  "decision_type": "choice",
  "options": ["movement", "animation", "audio"]
}
```

Treat confidence as evidence, not authority. Clients decide how to use results and
must keep deterministic safety, authorization, and correctness checks outside Laya.
The server never invokes tools, reads files, edits source, or executes commands.

