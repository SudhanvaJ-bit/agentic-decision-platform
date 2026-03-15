import json
import pytest
from app.agents.resume_intelligence import ResumeIntelligenceAgent
from app.agents.jd_deconstructor import JDDeconstructorAgent
from app.core.llm_provider import MockLLMProvider
from app.core.models import AgentStatus, WorkflowType
from app.core.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

STRONG_RESUME = """
John Smith | john@example.com | github.com/johnsmith

EXPERIENCE:
Senior Backend Engineer — TechCorp (2020–2024)
- Built FastAPI microservices handling 50,000 requests/sec
- Reduced PostgreSQL query latency by 60% through index optimization
- Led migration of 3 legacy Django apps to async FastAPI (Python 3.11)
- Deployed Kubernetes clusters on AWS EKS serving 2M daily active users

Backend Engineer — StartupXYZ (2018–2020)
- Developed REST APIs in Python/Flask for fintech platform
- Implemented Redis caching layer reducing DB load by 40%

SKILLS: Python, FastAPI, Django, PostgreSQL, Redis, Docker, Kubernetes, AWS
EDUCATION: B.Sc. Computer Science, State University, 2018
""".strip()

WEAK_RESUME = """
Alex Johnson | alex@email.com

EXPERIENCE:
Senior Synergy Engineer — VagueCorp (2023–2024, 3 months)
- Leveraged cross-functional paradigm shifts to drive digital transformation
- Worked on improving performance across various systems
- Led diverse team of stakeholders to deliver impactful solutions
- Interfaced with key decision makers to align strategic objectives

SKILLS: Python, leadership, synergy, innovation, blockchain, AI, cloud
""".strip()


def make_mock_resume_response(
    candidate_name="John Smith",
    experience_years=6,
    red_flags=None,
    strong_signals=None,
    missing_must_have=None,
    confidence=0.88,
) -> str:
    return json.dumps({
        "candidate_name": candidate_name,
        "total_experience_years": experience_years,
        "skills_extracted": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Kubernetes"],
        "red_flags": red_flags or [],
        "strong_signals": strong_signals or [
            "Quantified: 50,000 req/sec throughput",
            "Quantified: 60% latency reduction",
            "Clear progression from engineer to senior engineer",
        ],
        "jd_alignment": {
            "matched_must_have": ["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes"],
            "missing_must_have": missing_must_have or [],
            "matched_good_to_have": ["AWS"],
        },
        "reasoning": [
            "Candidate shows strong quantified achievements.",
            "All must-have skills matched from JD.",
        ],
        "confidence": confidence,
    })


def make_mock_jd_result() -> dict:
    """Simulate what JDDeconstructorAgent would have added to prior_results."""
    return {
        "agent_name": "jd_deconstructor",
        "status": "success",
        "summary": "JD deconstructed successfully.",
        "confidence": 0.92,
        "reasoning": ["Python required.", "Kubernetes required."],
        "flags": [],
        "data": {
            "must_have": ["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes"],
            "good_to_have": ["AWS", "Kafka"],
            "deal_breakers": ["Must be US-based"],
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_resume_intelligence_strong_candidate():
    """A good resume should get high confidence, low flags, matched skills."""
    mock_llm = MockLLMProvider(response=make_mock_resume_response())
    agent = ResumeIntelligenceAgent(llm=mock_llm)

    response = await agent.safe_run({
        "original_input": {"resume_text": STRONG_RESUME},
        "prior_results": [make_mock_jd_result()],
    })

    assert response.status == AgentStatus.SUCCESS
    assert response.confidence >= 0.7
    assert response.data["candidate_name"] == "John Smith"
    assert len(response.data["strong_signals"]) > 0
    assert "MISSING_MUST_HAVE_SKILLS" not in response.flags
    assert mock_llm.call_count == 1


async def test_resume_intelligence_weak_candidate():
    """A buzzword-heavy resume should surface HIGH_RED_FLAG_COUNT."""
    mock_llm = MockLLMProvider(response=make_mock_resume_response(
        candidate_name="Alex Johnson",
        experience_years=1,
        red_flags=[
            "buzzword: 'synergy'",
            "buzzword: 'paradigm shift'",
            "vague: 'improving performance'",
            "suspiciously short tenure: 3 months as Senior Engineer",
            "no quantified achievements",
        ],
        strong_signals=[],
        confidence=0.75,
    ))
    agent = ResumeIntelligenceAgent(llm=mock_llm)

    response = await agent.safe_run({
        "original_input": {"resume_text": WEAK_RESUME},
        "prior_results": [make_mock_jd_result()],
    })

    assert response.status == AgentStatus.SUCCESS
    assert "HIGH_RED_FLAG_COUNT" in response.flags
    assert "HUMAN_REVIEW_REQUIRED" in response.flags
    assert len(response.data["red_flags"]) >= 3


async def test_resume_intelligence_missing_must_have_skills():
    """Candidate missing required skills should get MISSING_MUST_HAVE_SKILLS flag."""
    mock_llm = MockLLMProvider(response=make_mock_resume_response(
        missing_must_have=["Kubernetes", "Docker"]
    ))
    agent = ResumeIntelligenceAgent(llm=mock_llm)

    response = await agent.safe_run({
        "original_input": {"resume_text": STRONG_RESUME},
        "prior_results": [make_mock_jd_result()],
    })

    assert "MISSING_MUST_HAVE_SKILLS" in response.flags
    assert "Kubernetes" in response.data["jd_alignment"]["missing_must_have"]


async def test_resume_intelligence_missing_input():
    """Missing resume_text should fail cleanly."""
    mock_llm = MockLLMProvider()
    agent = ResumeIntelligenceAgent(llm=mock_llm)

    response = await agent.safe_run({
        "original_input": {},
        "prior_results": [],
    })

    assert response.status == AgentStatus.FAILED
    assert "MISSING_INPUT" in response.flags
    assert mock_llm.call_count == 0


async def test_resume_reads_jd_context_from_prior_results():
    mock_llm = MockLLMProvider(response=make_mock_resume_response())
    agent = ResumeIntelligenceAgent(llm=mock_llm)

    response = await agent.safe_run({
        "original_input": {"resume_text": STRONG_RESUME},
        "prior_results": [make_mock_jd_result()],
    })

    assert response.status == AgentStatus.SUCCESS
    assert "jd_alignment" in response.data
    assert "matched_must_have" in response.data["jd_alignment"]


async def test_full_two_agent_hiring_pipeline():
    jd_response = json.dumps({
        "must_have": ["Python", "FastAPI", "PostgreSQL"],
        "good_to_have": ["AWS"],
        "deal_breakers": [],
        "reasoning": ["Python is required.", "FastAPI is required."],
        "confidence": 0.9,
    })

    resume_response = make_mock_resume_response()

    responses = [jd_response, resume_response]
    call_index = 0

    class SequencedMockLLM(MockLLMProvider):
        async def complete(self, system_prompt, user_prompt, **kwargs):
            nonlocal call_index
            response = responses[call_index % len(responses)]
            call_index += 1
            return response

    mock_llm = SequencedMockLLM()

    orchestrator = Orchestrator(
        workflow_type=WorkflowType.HIRING,
        agents=[
            JDDeconstructorAgent(llm=mock_llm),
            ResumeIntelligenceAgent(llm=mock_llm),
        ],
    )

    state = await orchestrator.run(input_payload={
        "jd_text": "Senior Python engineer needed with FastAPI and PostgreSQL skills...",
        "resume_text": STRONG_RESUME,
    })

    assert state.is_complete
    assert len(state.agent_results) == 2
    assert state.agent_results[0].agent_name == "jd_deconstructor"
    assert state.agent_results[1].agent_name == "resume_intelligence"
    # Resume agent should have received JD output in its context
    assert state.agent_results[1].data["jd_alignment"] is not None
