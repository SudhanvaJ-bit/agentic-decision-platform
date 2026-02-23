import uuid
import logging
from typing import Any

from app.core.base_agent import BaseAgent
from app.core.models import AgentResponse, AgentStatus, OrchestratorState, WorkflowType
from app.core.settings import settings

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = settings.confidence_threshold

HUMAN_ESCALATION_FLAGS = {"HUMAN_REVIEW_REQUIRED", "AGENT_CRASHED", "CRITICAL_RISK"}


class Orchestrator:
    def __init__(self, workflow_type: WorkflowType, agents: list[BaseAgent]):
        if not agents:
            raise ValueError("Orchestrator requires at least one agent.")

        self.workflow_type = workflow_type
        self.agents = agents

        logger.info(
            f"Orchestrator initialized | workflow={workflow_type} | "
            f"agents=[{', '.join(a.agent_name for a in agents)}]"
        )

    async def run(self, input_payload: dict[str, Any]) -> OrchestratorState:
        state = OrchestratorState(
            run_id=str(uuid.uuid4()),
            workflow_type=self.workflow_type,
            input_payload=input_payload,
        )

        logger.info(f"Pipeline START | run_id={state.run_id}")

        for index, agent in enumerate(self.agents):
            state.current_step = index

            logger.info(
                f"[Step {index + 1}/{len(self.agents)}] Running agent: {agent.agent_name}"
            )

            agent_input = self._build_agent_input(state)
            response: AgentResponse = await agent.safe_run(agent_input)

            state.agent_results.append(response)

            if self._should_escalate(response):
                state.requires_human_approval = True
                logger.warning(
                    f"[{agent.agent_name}] Escalation triggered. "
                    f"Flags: {response.flags}. Pipeline paused for human review."
                )
                break

            if response.status == AgentStatus.FAILED:
                logger.error(
                    f"[{agent.agent_name}] Agent failed. Halting pipeline."
                )
                break

        state.is_complete = True
        logger.info(
            f"Pipeline END | run_id={state.run_id} | "
            f"steps_completed={state.current_step + 1} | "
            f"requires_human_approval={state.requires_human_approval}"
        )

        return state

    def _build_agent_input(self, state: OrchestratorState) -> dict[str, Any]:
        return {
            "run_id": state.run_id,
            "workflow_type": state.workflow_type,
            "original_input": state.input_payload,
            "prior_results": [r.model_dump() for r in state.agent_results],
        }

    def _should_escalate(self, response: AgentResponse) -> bool:
        has_escalation_flag = bool(set(response.flags) & HUMAN_ESCALATION_FLAGS)
        is_low_confidence = response.confidence < CONFIDENCE_THRESHOLD

        if has_escalation_flag:
            logger.warning(f"Escalation flag detected: {set(response.flags) & HUMAN_ESCALATION_FLAGS}")
        if is_low_confidence:
            logger.warning(
                f"Low confidence detected: {response.confidence:.2f} < threshold {CONFIDENCE_THRESHOLD}"
            )

        return has_escalation_flag or is_low_confidence