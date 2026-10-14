"""LLM-as-judge: faithfulness (groundedness) and relevance of an answer.

Both judges are prompts in git and Langfuse (`prompts/judge-*.yaml`), run on
the same Azure GPT-4o as the answers. A judge only labels; the score is
computed here, so it is deterministic given the labels:

    faithfulness  supported claims / all claims, over the context blocks the
                  answer was generated from; `contradicted` is counted apart
                  (a claim the context contradicts, or Guidance presented as
                  a binding requirement)
    relevance     1-5 from the rubric; answered turns only (refusals are
                  scored by the refusal metrics)

The judges always use the `production` label, never the label under test, so
a staging edit of `answer` is never graded by a judge edited along with it.
A malformed or cut-off judge reply is recorded as `judge_error` on the turn;
it never stops the run. Known limitation: GPT-4o grading GPT-4o (same-model
bias), noted in the roadmap.
"""

import json
from dataclasses import dataclass

from src.generation.context import render_context
from src.generation.llm import LLMClient, LLMResult
from src.prompts.registry import PRODUCTION, PromptRegistry
from src.retrieval.models import ContextBlock

FAITHFULNESS = "judge-faithfulness"
RELEVANCE = "judge-relevance"
VERDICTS = {"supported", "unsupported", "contradicted"}


@dataclass
class Judgement:
    name: str
    score: float | None = None
    contradicted: int | None = None  # faithfulness only
    output: dict | None = None  # the parsed reply
    error: str | None = None
    prompt: dict | None = None  # ResolvedPrompt.trace()
    llm: LLMResult | None = None

    @property
    def cost_usd(self) -> float:
        return self.llm.cost_usd if self.llm else 0.0

    def trace(self) -> dict:
        return {
            "score": self.score,
            "contradicted": self.contradicted,
            "output": self.output,
            "error": self.error,
            "prompt": self.prompt,
            "llm": self.llm.trace() if self.llm else None,
        }


@dataclass
class Judge:
    llm: LLMClient
    registry: PromptRegistry
    label: str = PRODUCTION

    def faithfulness(self, question: str, context: list[ContextBlock], answer: str) -> Judgement:
        variables = {"question": question, "context": render_context(context), "answer": answer}
        j = self._call(FAITHFULNESS, variables)
        if j.output is None:
            return j
        claims = j.output.get("claims")
        if not isinstance(claims, list) or not claims:
            j.error = "no claims in the reply"
            return j
        verdicts = [c.get("verdict") if isinstance(c, dict) else None for c in claims]
        if bad := [v for v in verdicts if v not in VERDICTS]:
            j.error = f"unknown verdicts: {bad}"
            return j
        j.score = verdicts.count("supported") / len(verdicts)
        j.contradicted = verdicts.count("contradicted")
        return j

    def relevance(self, question: str, answer: str) -> Judgement:
        j = self._call(RELEVANCE, {"question": question, "answer": answer})
        if j.output is None:
            return j
        score = j.output.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            j.error = f"score must be an integer 1-5, got {score!r}"
            return j
        j.score = float(score)
        return j

    def _call(self, name: str, variables: dict) -> Judgement:
        prompt = self.registry.get(name, self.label)
        j = Judgement(name=name, prompt=prompt.trace())
        try:
            j.llm = self.llm.complete(prompt, variables, name=name)
        except Exception as e:  # an outage loses one judgement, not the run
            j.error = f"{type(e).__name__}: {e}"[:500]
            return j
        if j.llm.finish_reason != "stop":
            j.error = f"finish_reason {j.llm.finish_reason}"
            return j
        try:
            output = json.loads(j.llm.text)
        except json.JSONDecodeError as e:
            j.error = f"invalid JSON: {e}"
            return j
        if not isinstance(output, dict):
            j.error = "the reply is not a JSON object"
            return j
        j.output = output
        return j
