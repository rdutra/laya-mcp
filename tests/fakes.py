from __future__ import annotations

from typing import Any


class FakeTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        tokens = [index + 100 for index, _ in enumerate(text.split())]
        return {"input_ids": tokens}


class FakeAgent:
    def __init__(
        self,
        max_length: int = 96,
        *,
        confidence_by_id: dict[str, float] | None = None,
        noul_by_id: dict[str, float] | None = None,
        fail_on_calls: set[int] | None = None,
    ) -> None:
        self.tok = FakeTokenizer()
        self.cfg = {"max_len": 1024, "head_max_len": 256}
        self.shape = {
            "max_length": max_length,
            "max_options": 32,
            "batch_size": 1,
        }
        self.compute_units = "cpu_ne"
        self.predict_calls = 0
        self.confidence_by_id = confidence_by_id or {}
        self.noul_by_id = noul_by_id or {}
        self.fail_on_calls = fail_on_calls or set()

    def predict(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        del state
        self.predict_calls += 1
        if self.predict_calls in self.fail_on_calls:
            raise RuntimeError("simulated backend failure")
        answers: dict[str, dict[str, Any]] = {}
        for item_id, definition in questions.items():
            kind = definition["type"]
            base: dict[str, Any] = {
                "type": kind,
                "confidence": self.confidence_by_id.get(item_id, 0.8125),
                "action": {"act_probability": 0.9},
            }
            if kind == "noul":
                base["noul"] = self.noul_by_id.get(item_id, 0.8125)
            elif kind == "choice":
                options = definition["criteria"]
                base["choice"] = options[0]
                base["probabilities"] = {
                    option: (0.75 if index == 0 else 0.25 / (len(options) - 1))
                    for index, option in enumerate(options)
                }
            else:
                options = definition["criteria"]
                base["score"] = 1.25
                base["legend"] = {str(index): value for index, value in enumerate(options)}
                base["probabilities"] = {
                    str(index): 1.0 / len(options) for index in range(len(options))
                }
            answers[item_id] = base
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": 42 * len(questions), "output_tokens": 0},
        }
