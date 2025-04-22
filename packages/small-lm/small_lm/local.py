"""A local GGUF model through llama-cpp-python, with optional retrieval from a notes index."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from small_lm.retrieval import Index


@dataclass
class Generation:
    text: str
    first_token_logprobs: list[float]  # top-k log-probabilities of the first generated token


class LocalModel(Protocol):
    def generate(self, prompt: str, max_tokens: int = 256) -> Generation: ...


class Local:
    def __init__(self, model_path: str, index: str | None = None, n_ctx: int = 8192, n_threads: int | None = None,
                 top_k_context: int = 4):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError('running GGUF models needs: pip install "small-lm[run]"') from exc
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, logits_all=False, verbose=False)
        self.index = Index.open(index) if index else None
        self.top_k_context = top_k_context

    def prompt(self, question: str) -> str:
        if not self.index:
            return question
        notes = "\n".join(f"- {h.text}" for h in self.index.search(question, k=self.top_k_context))
        return f"Notes:\n{notes}\n\nUsing only the notes above, answer: {question}"

    def generate(self, prompt: str, max_tokens: int = 256) -> Generation:
        out = self.llm.create_chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens,
                                              temperature=0, logprobs=True, top_logprobs=5)
        choice = out["choices"][0]
        content = choice["logprobs"]["content"] if choice.get("logprobs") else []
        first = [t["logprob"] for t in content[0]["top_logprobs"]] if content else [0.0]
        return Generation(choice["message"]["content"], first)


def features(question: str, gen: Generation) -> list[float]:
    """What the router reads: the first token's top probability and entropy, and question length."""
    probs = [math.exp(lp) for lp in gen.first_token_logprobs]
    total = sum(probs) or 1.0
    norm = [p / total for p in probs]
    entropy = -sum(p * math.log(p) for p in norm if p > 0)
    return [max(probs), entropy, math.log1p(len(question.split()))]
