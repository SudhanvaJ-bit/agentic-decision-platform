from pydantic import BaseModel, Field
from app.agents.jd_deconstructor import JDDeconstructorAgent
from app.agents.resume_intelligence import ResumeIntelligenceAgent
from app.core.llm_provider import BaseLLMProvider, get_llm_provider
from app.core.models import OrchestratorState, WorkflowType
from app.core.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Input contract for the Hiring workflow
# ---------------------------------------------------------------------------

class HiringWorkflowInput(BaseModel):
    jd_text: str = Field(
        ...,
        min_length=50,
        description="The full text of the job description. Minimum 50 characters."
    )
    resume_text: str = Field(
        ...,
        min_length=50,
        description="The full text of the candidate's resume. Minimum 50 characters."
    )
    candidate_id: str = Field(
        default="unknown",
        description="Optional candidate identifier for audit trail."
    )

# ---------------------------------------------------------------------------
# Workflow factory function
# ---------------------------------------------------------------------------

async def run_hiring_pipeline(
    jd_text: str,
    resume_text: str,
    candidate_id: str = "unknown",
    llm: BaseLLMProvider | None = None,
) -> OrchestratorState:

    provider = llm or get_llm_provider()

    agents = [
        JDDeconstructorAgent(llm=provider),
        ResumeIntelligenceAgent(llm=provider),
    ]

    orchestrator = Orchestrator(
        workflow_type=WorkflowType.HIRING,
        agents=agents,
    )

    return await orchestrator.run(
        input_payload={
            "jd_text": jd_text,
            "resume_text": resume_text,
            "candidate_id": candidate_id,
        }
    )
