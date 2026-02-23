import time
import logging
from abc import ABC, abstractmethod
from typing import Any

from app.core.models import AgentResponse, AgentStatus

logger = logging.getLogger(__name__)


class BaseAgent(ABC):

    agent_name: str = "base_agent"

    @abstractmethod
    async def run(self, input_data: dict[str, Any]) -> AgentResponse:
        
        ...

    async def safe_run(self, input_data: dict[str, Any]) -> AgentResponse:
        
        start = time.monotonic()
        logger.info(f"[{self.agent_name}] Starting execution.")

        try:
            response = await self.run(input_data)

            
            if not isinstance(response, AgentResponse):
                raise TypeError(
                    f"Agent {self.agent_name} returned {type(response)} instead of AgentResponse."
                )

            elapsed = time.monotonic() - start
            logger.info(
                f"[{self.agent_name}] Completed in {elapsed:.3f}s | "
                f"status={response.status} | confidence={response.confidence:.2f}"
            )
            return response

        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(
                f"[{self.agent_name}] FAILED after {elapsed:.3f}s — {type(exc).__name__}: {exc}",
                exc_info=True,
            )
            
            return AgentResponse(
                agent_name=self.agent_name,
                status=AgentStatus.FAILED,
                summary=f"Agent encountered an unexpected error: {type(exc).__name__}",
                confidence=0.0,
                reasoning=["Agent raised an unhandled exception. See logs for full traceback."],
                flags=["AGENT_CRASHED", "HUMAN_REVIEW_REQUIRED"],
                data={"error": str(exc)},
            )