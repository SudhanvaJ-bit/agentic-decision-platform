from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.models import OrchestratorState, WorkflowType
from app.core.orchestrator import Orchestrator
from app.agents.dummy_agent import DummyAgent

router = APIRouter()

class RunWorkflowRequest(BaseModel):

    workflow_type: WorkflowType
    payload: dict[str, Any]

@router.get("/health", tags=["System"])
async def health_check() -> dict:
    return {"status": "ok", "service": "enterprise-agent-platform"}


@router.post(
    "/workflow/run",
    response_model=OrchestratorState,
    tags=["Workflow"],
    summary="Trigger a workflow run (smoke-test endpoint)",
    description=(
        "Runs the dummy agent pipeline to verify the orchestrator is wired correctly. "
        "Real hiring/procurement agents will be added in subsequent steps."
    ),
)
async def run_workflow(request: RunWorkflowRequest) -> OrchestratorState:
    agents = [DummyAgent()]

    orchestrator = Orchestrator(
        workflow_type=request.workflow_type,
        agents=agents,
    )

    try:
        state = await orchestrator.run(input_payload=request.payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return state