"""answer(question, retrieval) -> AnswerResult: a cited answer, or a refusal.

    1. gate      no context, or retrieval below the calibrated threshold:
                 refuse without calling the LLM
    2. generate  the `answer` prompt over the numbered context blocks (JSON out)
    3. validate  the model must be confident, and every provision it names must be
                 a context block it listed (citations.py)
    4. result    the answer (labelled when it rests only on Guidance), or the
                 `refusal` prompt with a one-sentence reason

A wrong or unsupported answer is worse than none, so every failed check is a
refusal and nothing is patched. LLM and API errors are raised instead: an outage
is not "no confident answer".
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import ValidationError

from src.generation.citations import AnswerDraft, FindCitations, Validation, check_citations
from src.generation.context import cite_as, render_context
from src.generation.llm import LLMClient, LLMResult
from src.observability.langfuse import get_langfuse
from src.prompts.registry import PromptRegistry
from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

ANSWER_PROMPT, REFUSAL_PROMPT = "answer", "refusal"
NO_HISTORY = "(none)"

RefusalCode = Literal[
    "no_context",
    "low_confidence",
    "invalid_output",
    "model_not_confident",
    "citation_check_failed",
]
# The sentence each refusal gives as its reason (the `refusal` prompt's {{reason}}).
REASONS: dict[str, str] = {
    "no_context": "I couldn't find any provisions related to this question.",
    "low_confidence": "The closest provisions I found don't clearly address this question.",
    "invalid_output": "I couldn't produce a complete answer this time.",
    "model_not_confident": "The provisions I found only partly cover this question.",
    "citation_check_failed": (
        "I couldn't verify that a draft answer was fully supported by the provisions it "
        "cited, so I haven't given it."
    ),
}
GUIDANCE_ONLY_NOTE = (
    "Note: this answer rests only on non-binding DFSA Guidance, not on a binding Rule."
)


@dataclass
class CitedProvision:
    context_id: int
    chunk_id: str
    key: str
    citation: str  # as displayed: "COB 3.3.2", "COB 3.3.2 Guidance"
    label: str  # "Rule (binding)"
    binding: bool
    heading: str


@dataclass
class Refusal:
    code: RefusalCode
    reason: str  # the sentence shown to the user


@dataclass
class AnswerResult:
    question: str
    status: Literal["answered", "refused"]
    text: str  # what the user sees: the answer, or the refusal
    cited: list[CitedProvision] = field(default_factory=list)
    guidance_only: bool = False
    refusal: Refusal | None = None
    gate: str = ""  # passed | uncalibrated | no_context | low_confidence
    violations: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)  # provisions the answer text names
    relayed: list[str] = field(default_factory=list)  # ... as referred to by a cited block
    draft: dict | str | None = None  # the model's output, parsed if it parsed
    prompts: list[dict] = field(default_factory=list)  # ResolvedPrompt.trace() of each used
    llm: LLMResult | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def refused(self) -> bool:
        return self.status == "refused"

    @property
    def cost_usd(self) -> float:
        return self.llm.cost_usd if self.llm else 0.0

    def trace(self) -> dict:
        """Everything Phase G stores about the answer step, as plain JSON."""
        return {
            "status": self.status,
            "refusal": asdict(self.refusal) if self.refusal else None,
            "gate": self.gate,
            "cited": [asdict(c) for c in self.cited],
            "guidance_only": self.guidance_only,
            "violations": self.violations,
            "mentions": self.mentions,
            "relayed": self.relayed,
            "draft": self.draft,
            "prompts": self.prompts,
            "llm": self.llm.trace() if self.llm else None,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
        }


class Answerer:
    def __init__(self, llm: LLMClient, registry: PromptRegistry, find_citations: FindCitations):
        """`find_citations` reads provision numbers out of text: the retriever's
        CitationRouter.find, so answers and questions are parsed alike."""
        self.llm = llm
        self.registry = registry
        self.find_citations = find_citations

    def answer(
        self,
        retrieval: RetrievalResult,
        *,
        history: str = NO_HISTORY,
        label: str | None = None,
    ) -> AnswerResult:
        """`label` picks the prompt version (default: production; evals use staging)."""
        start = time.perf_counter()
        langfuse = get_langfuse()
        with langfuse.start_as_current_observation(
            name="answer", as_type="chain", input={"question": retrieval.question}
        ) as span:
            result = self._answer(retrieval, history, label)
            result.timings_ms["total"] = (time.perf_counter() - start) * 1000
            span.update(
                output={
                    "status": result.status,
                    "text": result.text,
                    "cited": [c.citation for c in result.cited],
                },
                metadata={
                    "refusal": result.refusal and asdict(result.refusal),
                    "gate": result.gate,
                    "guidance_only": result.guidance_only,
                },
            )
        logger.info(
            "answer: %s%s, %d cited, $%.6f, %.0f ms",
            result.status,
            f" ({result.refusal.code})" if result.refusal else "",
            len(result.cited),
            result.cost_usd,
            result.timings_ms["total"],
        )
        return result

    def _answer(self, retrieval: RetrievalResult, history: str, label: str | None) -> AnswerResult:
        result = AnswerResult(question=retrieval.question, status="answered", text="")

        # 1. gate: refuse without an LLM call
        if not retrieval.context:
            result.gate = "no_context"
            return self._refuse(result, retrieval, "no_context", label)
        if retrieval.confident is False:
            result.gate = "low_confidence"
            return self._refuse(result, retrieval, "low_confidence", label)
        # None: no calibrated threshold yet (Phase H), or no reranker.
        result.gate = "passed" if retrieval.confident else "uncalibrated"

        # 2. generate
        prompt = self.registry.get(ANSWER_PROMPT, label)
        result.prompts.append(prompt.trace())
        variables = {
            "history": history or NO_HISTORY,
            "context": render_context(retrieval.context),
            "question": retrieval.question,
        }
        t = time.perf_counter()
        result.llm = self.llm.complete(prompt, variables, name="generate-answer")
        result.timings_ms["llm"] = (time.perf_counter() - t) * 1000
        result.draft = result.llm.text

        if result.llm.finish_reason != "stop":  # cut off, or a content filter
            result.violations.append(f"finish_reason {result.llm.finish_reason}")
            return self._refuse(result, retrieval, "invalid_output", label)
        try:
            draft = AnswerDraft.model_validate(json.loads(result.llm.text))
        except (json.JSONDecodeError, ValidationError) as e:
            result.violations.append(f"output is not the expected JSON: {str(e)[:200]}")
            return self._refuse(result, retrieval, "invalid_output", label)
        result.draft = draft.model_dump()

        # 3. validate
        validation = self._validate(draft, retrieval)
        result.violations += validation.violations
        result.mentions = validation.mentions
        result.relayed = validation.relayed
        if not draft.confident:
            return self._refuse(result, retrieval, "model_not_confident", label)
        if not validation.ok:
            return self._refuse(result, retrieval, "citation_check_failed", label)

        # 4. the answer
        result.cited = [
            CitedProvision(
                context_id=b.n,
                chunk_id=b.chunk_id,
                key=b.key,
                citation=cite_as(b),
                label=b.label,
                binding=b.binding,
                heading=b.heading,
            )
            for b in validation.cited
        ]
        result.guidance_only = validation.guidance_only
        text = draft.answer.strip()
        result.text = f"{GUIDANCE_ONLY_NOTE}\n\n{text}" if result.guidance_only else text
        return result

    def _validate(self, draft: AnswerDraft, retrieval: RetrievalResult) -> Validation:
        with get_langfuse().start_as_current_observation(
            name="validate-citations",
            as_type="guardrail",
            input={
                "citations": [c.model_dump() for c in draft.citations],
                "confident": draft.confident,
            },
        ) as obs:
            validation = check_citations(draft, retrieval.context, self.find_citations)
            obs.update(
                output={
                    "ok": validation.ok,
                    "violations": validation.violations,
                    "mentions": validation.mentions,
                    "relayed": validation.relayed,
                    "guidance_only": validation.guidance_only,
                },
                level="WARNING" if not validation.ok else "DEFAULT",
            )
        return validation

    def _refuse(
        self,
        result: AnswerResult,
        retrieval: RetrievalResult,
        code: RefusalCode,
        label: str | None,
    ) -> AnswerResult:
        reason = REASONS[code]
        if missing := retrieval.citations.unresolved:
            names = ", ".join(missing)
            verb = "is" if len(missing) == 1 else "are"
            reason += f" {names} {verb} not in the documents I can search."
        prompt = self.registry.get(REFUSAL_PROMPT, label)
        result.prompts.append(prompt.trace())
        result.status = "refused"
        result.refusal = Refusal(code=code, reason=reason)
        result.text = prompt.compile(reason=reason).strip()
        result.cited = []
        result.guidance_only = False
        return result
