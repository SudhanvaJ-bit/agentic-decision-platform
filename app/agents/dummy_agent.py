from typing import Any

from app.core.base_agent import BaseAgent
from app.core.models import AgentResponse, AgentStatus


class DummyAgent(BaseAgent):
    agent_name = "dummy_agent"

    async def run(self, input_data: dict[str, Any]) -> AgentResponse:
        run_id = input_data.get("run_id", "unknown")
        workflow = input_data.get("workflow_type", "unknown")
        prior_count = len(input_data.get("prior_results", []))

        return AgentResponse(
            agent_name=self.agent_name,
            status=AgentStatus.SUCCESS,
            summary=(
                f"DummyAgent executed successfully for run_id={run_id}. "
                f"This is a smoke-test agent with no real logic."
            ),
            confidence=1.0,
            reasoning=[
                f"Received input for workflow type: {workflow}.",
                f"Found {prior_count} prior agent result(s) in context.",
                "No processing logic required — this is a passthrough agent.",
            ],
            flags=[],
            data={
                "echo": {
                    "run_id": run_id,
                    "workflow_type": str(workflow),
                    "prior_results_count": prior_count,
                }
            },
        )
    
class AlwaysFailAgent(BaseAgent):
    agent_name = "always_fail_agent"

    async def run(self, input_data: dict[str, Any]) -> AgentResponse:
        raise RuntimeError("Intentional failure for pipeline resilience testing.")