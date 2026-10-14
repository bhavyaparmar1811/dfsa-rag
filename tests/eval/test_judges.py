import pytest

from src.eval.judges import Judge
from src.generation.llm import LLMClient
from tests.generation.conftest import BLOCKS, FakeAzure, azure_settings, git_registry


def judge(*replies, finish_reason="stop") -> tuple[Judge, FakeAzure]:
    fake = FakeAzure(*replies, finish_reason=finish_reason)
    return Judge(LLMClient(azure_settings(), client=fake), git_registry()), fake


def claims(*verdicts):
    return {"claims": [{"claim": f"c{i}", "verdict": v, "context_ids": [1]}
                       for i, v in enumerate(verdicts)]}  # fmt: skip


def test_faithfulness_scores_supported_share_and_counts_contradictions():
    j, fake = judge(claims("supported", "supported", "unsupported", "contradicted"))
    result = j.faithfulness("Q?", BLOCKS, "An answer (COB 3.3.2).")
    assert result.score == 0.5 and result.contradicted == 1 and result.error is None
    assert result.cost_usd > 0 and result.prompt["name"] == "judge-faithfulness"
    user = fake.calls[0]["messages"][-1]["content"]
    assert "[1] " in user and "An answer (COB 3.3.2)." in user  # the rendered context blocks


def test_relevance_score():
    j, _ = judge({"reasoning": "covers it", "score": 5})
    assert j.relevance("Q?", "A.").score == 5.0


@pytest.mark.parametrize(
    "reply, error",
    [
        ("not json", "invalid JSON"),
        ("[1, 2]", "not a JSON object"),
        ({"claims": []}, "no claims"),
        (claims("supported", "maybe"), "unknown verdicts"),
    ],
)
def test_faithfulness_errors_are_recorded(reply, error):
    j, _ = judge(reply)
    result = j.faithfulness("Q?", BLOCKS, "A.")
    assert result.score is None and error in result.error


@pytest.mark.parametrize("score", [0, 6, 4.5, "4", True, None])
def test_relevance_rejects_bad_scores(score):
    j, _ = judge({"reasoning": "x", "score": score})
    result = j.relevance("Q?", "A.")
    assert result.score is None and "integer 1-5" in result.error


def test_cut_off_reply_and_outage_do_not_raise():
    j, _ = judge({"reasoning": "x", "score": 4}, finish_reason="length")
    assert j.relevance("Q?", "A.").error == "finish_reason length"
    j, _ = judge(RuntimeError("azure down"))
    assert "azure down" in j.relevance("Q?", "A.").error
