import pytest
from app.core.models import AgentResponse, AgentStatus, WorkflowType
from app.core.orchestrator import Orchestrator
from app.agents.dummy_agent import DummyAgent, AlwaysFailAgent


def test_agent_response_rejects_empty_reasoning():
    with pytest.raises(Exception):
        AgentResponse(
            agent_name="test",
            summary="something",
            confidence=0.9,
            reasoning=[],
        )

def test_agent_response_rejects_blank_summary():
    with pytest.raises(Exception):
        AgentResponse(
            agent_name="test",
            summary="   ",
            confidence=0.9,
            reasoning=["Did something."],
        )

def test_valid_agent_response_construction():
    response = AgentResponse(
        agent_name="test_agent",
        summary="All checks passed.",
        confidence=0.85,
        reasoning=["Step 1: checked input.", "Step 2: validated output."],
        flags=["LOW_CONFIDENCE"],
        data={"score": 42},
    )
    assert response.agent_name == "test_agent"
    assert response.status == AgentStatus.SUCCESS
    assert len(response.reasoning) == 2


async def test_dummy_agent_runs_successfully():
    orchestrator = Orchestrator(
        workflow_type=WorkflowType.HIRING,
        agents=[DummyAgent()],
    )
    state = await orchestrator.run(input_payload={"candidate_name": "Alice"})
    assert state.is_complete is True
    assert len(state.agent_results) == 1
    assert state.agent_results[0].agent_name == "dummy_agent"
    assert state.agent_results[0].status == AgentStatus.SUCCESS
    assert state.requires_human_approval is False


async def test_always_fail_agent_does_not_crash_pipeline():
    orchestrator = Orchestrator(
        workflow_type=WorkflowType.PROCUREMENT,
        agents=[AlwaysFailAgent()],
    )
    state = await orchestrator.run(input_payload={"item": "laptops"})
    assert state.is_complete is True
    assert state.agent_results[0].status == AgentStatus.FAILED
    assert "AGENT_CRASHED" in state.agent_results[0].flags
    assert state.requires_human_approval is True


async def test_multi_agent_pipeline_passes_context():
    orchestrator = Orchestrator(
        workflow_type=WorkflowType.HIRING,
        agents=[DummyAgent(), DummyAgent()],
    )
    state = await orchestrator.run(input_payload={"test": True})
    assert len(state.agent_results) == 2
    assert state.agent_results[1].data["echo"]["prior_results_count"] == 1
