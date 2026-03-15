import json
import logging
from typing import Any

from app.core.base_agent import BaseAgent
from app.core.llm_provider import BaseLLMProvider, get_llm_provider
from app.core.models import AgentResponse, AgentStatus

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert HR analyst and talent acquisition specialist.
Your job is to analyze job descriptions and extract structured requirements.

You MUST respond with ONLY valid JSON. No explanation, no markdown, no preamble.
Any deviation from JSON will break the hiring pipeline.

Output this exact structure:
{
  "must_have": ["requirement 1", "requirement 2"],
  "good_to_have": ["nice skill 1", "nice skill 2"],
  "deal_breakers": ["disqualifier 1", "disqualifier 2"],
  "reasoning": ["why you classified X as must_have", "..."],
  "confidence": 0.95
}

Rules:
- must_have: Only include skills/experience explicitly stated as required or mandatory.
- good_to_have: Include preferred, nice-to-have, or bonus qualifications.
- deal_breakers: Include only explicitly stated disqualifiers (e.g., "must not have criminal record", "no remote candidates").
- If no deal_breakers are stated, return an empty list.
- confidence: Your confidence in the accuracy of this extraction (0.0 to 1.0).
- reasoning: At least 2 bullet points explaining your classification decisions.
""".strip()


def build_user_prompt(jd_text: str) -> str:
    return f"""
Analyze this job description and extract the structured requirements:

--- JOB DESCRIPTION START ---
{jd_text}
--- JOB DESCRIPTION END ---

Respond with valid JSON only.
""".strip()

class JDDeconstructorAgent(BaseAgent):
    agent_name = "jd_deconstructor"

    def __init__(self, llm: BaseLLMProvider | None = None):
        # If no LLM is injected, use the factory (reads from .env)
        self.llm = llm or get_llm_provider()

    async def run(self, input_data: dict[str, Any]) -> AgentResponse:
        # --- 1. Extract and validate input ---
        jd_text = input_data.get("original_input", {}).get("jd_text", "").strip()

        if not jd_text:
            return AgentResponse(
                agent_name=self.agent_name,
                status=AgentStatus.FAILED,
                summary="No job description text was provided.",
                confidence=0.0,
                reasoning=["Input payload missing required field: 'jd_text'."],
                flags=["MISSING_INPUT", "HUMAN_REVIEW_REQUIRED"],
            )

        # --- 2. Call the LLM ---
        logger.info(f"[{self.agent_name}] Sending JD to LLM ({len(jd_text)} chars)")

        raw_response = await self.llm.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(jd_text),
            temperature=0.1,   # Very low: we want consistent, not creative
        )

        # --- 3. Parse and validate the LLM's JSON response ---
        parsed = self._parse_llm_response(raw_response)
        if parsed is None:
            return AgentResponse(
                agent_name=self.agent_name,
                status=AgentStatus.FAILED,
                summary="LLM returned malformed JSON. Cannot proceed with JD analysis.",
                confidence=0.0,
                reasoning=[
                    "LLM response could not be parsed as valid JSON.",
                    f"Raw response preview: {raw_response[:200]}",
                ],
                flags=["LLM_PARSE_ERROR", "HUMAN_REVIEW_REQUIRED"],
                data={"raw_response": raw_response},
            )

        # --- 4. Extract fields with safe defaults ---
        must_have    = parsed.get("must_have", [])
        good_to_have = parsed.get("good_to_have", [])
        deal_breakers = parsed.get("deal_breakers", [])
        reasoning    = parsed.get("reasoning", ["LLM provided no reasoning."])
        confidence   = float(parsed.get("confidence", 0.7))

        # --- 5. Add flags if output looks suspicious ---
        flags = []
        if not must_have:
            flags.append("NO_MUST_HAVE_REQUIREMENTS_FOUND")
        if confidence < 0.6:
            flags.append("LOW_CONFIDENCE")

        # --- 6. Build and return the structured response ---
        return AgentResponse(
            agent_name=self.agent_name,
            status=AgentStatus.SUCCESS,
            summary=(
                f"JD successfully deconstructed. "
                f"Found {len(must_have)} must-have requirements, "
                f"{len(good_to_have)} good-to-have skills, "
                f"and {len(deal_breakers)} deal-breakers."
            ),
            confidence=confidence,
            reasoning=reasoning if isinstance(reasoning, list) else [str(reasoning)],
            flags=flags,
            data={
                "must_have": must_have,
                "good_to_have": good_to_have,
                "deal_breakers": deal_breakers,
            },
        )

    def _parse_llm_response(self, raw: str) -> dict | None:
        try:
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])  # Remove first and last line

            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            logger.error(f"[{self.agent_name}] JSON parse failed: {e}")
            return None