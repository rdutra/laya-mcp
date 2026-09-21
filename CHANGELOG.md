# Changelog

## 0.1.0 — unreleased

This release candidate provides a client-independent MCP server for local,
bounded decisions with Laya-CoreML on Apple Silicon.

Highlights:

- stable `info`, `decide`, `batch_decide`, and experimental `filter` tools;
- lazy model initialization with one resident model per process;
- static capability metadata before model load for known official models;
- serialized Core ML inference, deterministic token preflight, and explicit failure
  handling;
- the ANE W8 model as the default after limited local compatibility/quality checks;
- documented evaluation results, including the finding that automatic Codex token
  savings and silent context exclusion are not demonstrated.

This project has not been published or tagged by this repository change.
