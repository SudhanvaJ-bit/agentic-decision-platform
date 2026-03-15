import json
import pytest

from app.agents.jd_deconstructor import JDDeconstructorAgent
from app.core.llm_provider import MockLLMProvider
from app.core.models import AgentStatus
from app.core.orchestrator import Orchestrator
from app.core.models import WorkflowType


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLE_JD = """
We are looking for a Senior Backend Engineer to join our platform team.

Required:
- 5+ years of Python experience
- Strong knowledge of FastAPI or Django REST Framework
- Experience with PostgreSQL and Redis
- Proficiency with Docker and Kubernetes
- Understanding of REST API design principles

Nice to have:
- Experience with AWS or GCP
- Knowledge of message queues (Kafka, RabbitMQ)
- Open source contributions

Candidates must be based in the US. No remote candidates outside the US will be considered.
""".strip()


def make_mock_jd_response(
    must_have=None,
    good_to_have=None,
    deal_breakers=None,
    confidence=0.92,
) -> str:
    """Helper: build a canned LLM response JSON string."""
    return json.dumps({
        "must_have": must_have or ["Python 5+ years", "FastAPI", "PostgreSQL", "Docker", "Kubernetes"],
        "good_to_have": good_to_have or ["AWS", "Kafka", "Open source contributions"],
        "deal_breakers": deal_breakers or ["Must be US-based"],
        "reasoning": [
            "Python, FastAPI, and PostgreSQL are explicitly listed as required.",
            "AWS and Kafka are in the nice-to-have section.",
            "US location requirement is an explicit disqualifier for overseas candidates.",
        ],
        "confidence": confidence,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_jd_deconstructor_happy_path():
    """Standard case: valid JD produces structured output."""
    mock_llm = MockLLMProvider(response=make_mock_jd_response())
    agent = JDDeconstructorAgent(llm=mock_llm)

    response = await agent.safe_run({
        "original_input": {"jd_text": SAMPLE_JD},
        "prior_results": [],
    })

    assert response.status == AgentStatus.SUCCESS
    assert response.agent_name == "jd_deconstructor"
    assert response.confidence > 0.5
    assert "must_have" in response.data
    assert "good_to_have" in response.data
    assert "deal_breakers" in response.data
    assert len(response.data["must_have"]) > 0
    assert mock_llm.call_count == 1  # LLM called exactly once


async def test_jd_deconstructor_missing_input():
    """If jd_text is missing, agent should fail gracefully — not crash."""
    mock_llm = MockLLMProvider()
    agent = JDDeconstructorAgent(llm=mock_llm)

    response = await agent.safe_run({
        "original_input": {},   # No jd_text
        "prior_results": [],
    })

    assert response.status == AgentStatus.FAILED
    assert "MISSING_INPUT" in response.flags
    assert mock_llm.call_count == 0  # LLM should NOT be called if input is missing


async def test_jd_deconstructor_handles_malformed_llm_response():
    """If LLM returns garbage instead of JSON, agent fails gracefully."""
    mock_llm = MockLLMProvider(response="Sorry, I cannot help with that.")
    agent = JDDeconstructorAgent(llm=mock_llm)

    response = await agent.safe_run({
        "original_input": {"jd_text": SAMPLE_JD},
        "prior_results": [],
    })

    assert response.status == AgentStatus.FAILED
    assert "LLM_PARSE_ERROR" in response.flags
    assert response.confidence == 0.0


async def test_jd_deconstructor_handles_markdown_wrapped_json():
    """LLMs sometimes wrap JSON in ```json ... ``` even when told not to."""
    wrapped_response = "```json\n" + make_mock_jd_response() + "\n```"
    mock_llm = MockLLMProvider(response=wrapped_response)
    agent = JDDeconstructorAgent(llm=mock_llm)

    response = await agent.safe_run({
        "original_input": {"jd_text": SAMPLE_JD},
        "prior_results": [],
    })

    # Should parse correctly despite the markdown wrapper
    assert response.status == AgentStatus.SUCCESS
    assert len(response.data["must_have"]) > 0


async def test_jd_deconstructor_flags_low_confidence():
    """Low confidence from LLM should be surfaced as a flag."""
    mock_llm = MockLLMProvider(response=make_mock_jd_response(confidence=0.4))
    agent = JDDeconstructorAgent(llm=mock_llm)

    response = await agent.safe_run({
        "original_input": {"jd_text": SAMPLE_JD},
        "prior_results": [],
    })

    assert "LOW_CONFIDENCE" in response.flags


async def test_jd_deconstructor_in_full_pipeline():
    """Run through the Orchestrator to verify pipeline wiring."""
    mock_llm = MockLLMProvider(response=make_mock_jd_response())
    orchestrator = Orchestrator(
        workflow_type=WorkflowType.HIRING,
        agents=[JDDeconstructorAgent(llm=mock_llm)],
    )

    state = await orchestrator.run(input_payload={"jd_text": SAMPLE_JD})

    assert state.is_complete
    assert len(state.agent_results) == 1
    assert state.agent_results[0].agent_name == "jd_deconstructor"
    assert state.agent_results[0].data["must_have"]
