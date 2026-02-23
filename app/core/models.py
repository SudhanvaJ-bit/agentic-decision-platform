from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class WorkflowType(str, Enum):
    HIRING = "hiring"
    PROCUREMENT = "procurement"


class AgentStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"  

class AgentResponse(BaseModel):
    agent_name: str = Field(
        ...,
        description="Unique identifier for the agent that produced this response."
    )
    status: AgentStatus = Field(
        default=AgentStatus.SUCCESS,
        description="Did this agent complete, fail, or get intentionally skipped?"
    )
    summary: str = Field(
        ...,
        description="One or two sentence human-readable summary of what the agent concluded."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How confident is the agent in its output? "
            "0.0 = pure guess, 1.0 = certain. "
            "The Decision Engine uses this to weight verdicts."
        )
    )
    reasoning: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered list of reasoning steps that led to the summary. "
            "Must have at least one entry — no reasoning means no validity."
        )
    )
    flags: list[str] = Field(
        default_factory=list,
        description=(
            "Warnings or concerns the agent wants to surface. "
            "E.g. 'LOW_CONFIDENCE', 'MISSING_DATA', 'HUMAN_REVIEW_REQUIRED'."
        )
    )
    data: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Structured payload specific to this agent's domain. "
            "E.g., extracted skills, vendor scores, risk ratings."
        )
    )

    @field_validator("reasoning")
    @classmethod
    def reasoning_must_not_be_empty_strings(cls, v: list[str]) -> list[str]:
        cleaned = [r.strip() for r in v if r.strip()]
        if not cleaned:
            raise ValueError(
                "reasoning list must contain at least one non-empty string. "
                "An agent that cannot explain itself is an invalid agent."
            )
        return cleaned

    @field_validator("summary")
    @classmethod
    def summary_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("summary cannot be blank.")
        return v.strip()


class OrchestratorState(BaseModel):
    run_id: str = Field(..., description="Unique ID for this pipeline execution.")
    workflow_type: WorkflowType
    input_payload: dict[str, Any] = Field(
        ...,
        description="The raw input that triggered this workflow run."
    )
    agent_results: list[AgentResponse] = Field(
        default_factory=list,
        description="Ordered list of results as agents complete."
    )
    current_step: int = Field(
        default=0,
        description="Index of the agent currently executing."
    )
    is_complete: bool = Field(default=False)
    requires_human_approval: bool = Field(
        default=False,
        description="Set to True by any agent that raises a critical flag."
    )
    final_decision: Optional[str] = Field(
        default=None,
        description="The terminal verdict produced by the Decision Engine."
    )