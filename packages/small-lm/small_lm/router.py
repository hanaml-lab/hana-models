"""Answer locally when the local model is likely right; otherwise escalate or abstain."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

from small_lm.calibrate import RouterModel
from small_lm.local import LocalModel, features


@dataclass
class Answer:
    text: str
    route: str  # local | escalated | abstained
    confidence: float


class Router:
    def __init__(self, model: RouterModel, escalate_to: str | None = None, coverage: float = 0.8,
                 on_low_confidence: str = "escalate", escalate_model: str = "default"):
        if on_low_confidence not in {"escalate", "abstain"}:
            raise ValueError("on_low_confidence is escalate or abstain")
        if on_low_confidence == "escalate" and not escalate_to:
            raise ValueError("escalation needs escalate_to")
        self.model, self.escalate_to, self.coverage = model, escalate_to, coverage
        self.on_low, self.escalate_model = on_low_confidence, escalate_model

    @classmethod
    def load(cls, path: str, **kw) -> Router:
        return cls(RouterModel.load(path), **kw)

    def _escalate(self, prompt: str) -> str:
        body = {"model": self.escalate_model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(self.escalate_to.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {os.environ.get('ESCALATE_API_KEY', '')}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]

    def ask(self, local: LocalModel, question: str) -> Answer:
        prompt = local.prompt(question) if hasattr(local, "prompt") else question
        gen = local.generate(prompt)
        conf = float(self.model.confidence(features(question, gen))[0])
        if conf >= self.model.threshold(self.coverage):
            return Answer(gen.text, "local", conf)
        if self.on_low == "abstain":
            return Answer("I am not confident enough to answer this on the device.", "abstained", conf)
        return Answer(self._escalate(prompt), "escalated", conf)
