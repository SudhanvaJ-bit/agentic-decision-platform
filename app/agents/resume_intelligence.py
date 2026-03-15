import json
import logging
from typing import Any

from app.core.base_agent import BaseAgent
from app.core.llm_provider import BaseLLMProvider, get_llm_provider
from app.core.models import AgentResponse, AgentStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a senior technical recruiter with expertise in detecting resume quality.
Your job is to analyze resumes critically and objectively.

You MUST respond with ONLY valid JSON. No markdown, no explanation, no preamble.

Output this exact structure:
{
  "candidate_name": "extracted name or 'Unknown'",
  "total_experience_years": 5,
  "skills_extracted": ["Python", "FastAPI", "PostgreSQL"],
  "red_flags": ["vague claim: 'improved performance'", "buzzword: 'synergy'"],
  "strong_signals": ["specific project: built X that handled Y load", "quantified: reduced latency by 40%"],
  "jd_alignment": {
    "matched_must_have": ["Python", "REST APIs"],
    "missing_must_have": ["Kubernetes"],
    "matched_good_to_have": ["Docker"]
  },
  "reasoning": ["reason 1", "reason 2"],
  "confidence": 0.85
}

Red flag detection rules:
- Flag buzzwords with no measurable outcome (synergy, leverage, paradigm, rockstar)
- Flag vague claims without numbers (improved performance, led team, worked on)
- Flag projects with no tech stack, no scale, no outcome
- Flag suspiciously short tenures with senior titles (3 months as CTO)

Strong signal rules:
- Quantified achievements (reduced X by Y%, handled Z requests/sec)
- Named technologies with real-world context
- Open source contributions or GitHub links
- Progression from junior to senior roles over time
""".strip()


def build_user_prompt(resume_text: str, must_have_skills: list[str]) -> str:
    must_have_str = "\n".join(f"- {s}" for s in must_have_skills) if must_have_skills else "- Not provided"
    return f"""
Analyze this resume. The job requires these must-have skills:
{must_have_str}

--- RESUME START ---
{resume_text}
--- RESUME END ---

Respond with valid JSON only.
""".strip()


# ---------------------------------------------------------------------------
# The Agent
# ---------------------------------------------------------------------------

class ResumeIntelligenceAgent(BaseAgent):
    agent_name = "resume_intelligence"

    def __init__(self, llm: BaseLLMProvider | None = None):
        self.llm = llm or get_llm_provider()

    async def run(self, input_data: dict[str, Any]) -> AgentResponse:
        # --- 1. Extract resume text ---
        resume_text = input_data.get("original_input", {}).get("resume_text", "").strip()

        if not resume_text:
            return AgentResponse(
                agent_name=self.agent_name,
                status=AgentStatus.FAILED,
                summary="No resume text was provided in the input payload.",
                confidence=0.0,
                reasoning=["Input payload missing required field: 'resume_text'."],
                flags=["MISSING_INPUT", "HUMAN_REVIEW_REQUIRED"],
            )

        # --- 2. Pull must_have from JD Deconstructor's output if available ---
        must_have_skills = self._extract_must_have_from_prior(
            input_data.get("prior_results", [])
        )

        if not must_have_skills:
            logger.warning(
                f"[{self.agent_name}] No JD Deconstructor output found in prior_results. "
                "Resume will be analysed without JD context."
            )

        # --- 3. Call LLM ---
        logger.info(f"[{self.agent_name}] Analysing resume ({len(resume_text)} chars)")

        raw_response = await self.llm.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(resume_text, must_have_skills),
            temperature=0.1,
        )

        # --- 4. Parse response ---
        parsed = self._parse_llm_response(raw_response)
        if parsed is None:
            return AgentResponse(
                agent_name=self.agent_name,
                status=AgentStatus.FAILED,
                summary="LLM returned malformed JSON during resume analysis.",
                confidence=0.0,
                reasoning=["JSON parse failed on LLM response."],
                flags=["LLM_PARSE_ERROR", "HUMAN_REVIEW_REQUIRED"],
                data={"raw_response": raw_response},
            )

        # --- 5. Extract fields ---
        red_flags      = parsed.get("red_flags", [])
        strong_signals = parsed.get("strong_signals", [])
        jd_alignment   = parsed.get("jd_alignment", {})
        missing        = jd_alignment.get("missing_must_have", [])
        reasoning      = parsed.get("reasoning", ["No reasoning provided."])
        confidence     = float(parsed.get("confidence", 0.7))

        # --- 6. Compute flags ---
        flags = []
        if len(red_flags) >= 3:
            flags.append("HIGH_RED_FLAG_COUNT")
        if missing:
            flags.append(f"MISSING_MUST_HAVE_SKILLS")
        if confidence < 0.6:
            flags.append("LOW_CONFIDENCE")
        if len(red_flags) >= 5:
            # Too many red flags — this candidate warrants human review
            flags.append("HUMAN_REVIEW_REQUIRED")

        # --- 7. Build summary ---
        summary = (
            f"Resume analysed for {parsed.get('candidate_name', 'Unknown')}. "
            f"{parsed.get('total_experience_years', '?')} years experience. "
            f"{len(red_flags)} red flag(s), {len(strong_signals)} strong signal(s). "
            f"{len(missing)} must-have skill(s) missing from JD requirements."
        )

        return AgentResponse(
            agent_name=self.agent_name,
            status=AgentStatus.SUCCESS,
            summary=summary,
            confidence=confidence,
            reasoning=reasoning if isinstance(reasoning, list) else [str(reasoning)],
            flags=flags,
            data={
                "candidate_name": parsed.get("candidate_name", "Unknown"),
                "total_experience_years": parsed.get("total_experience_years", 0),
                "skills_extracted": parsed.get("skills_extracted", []),
                "red_flags": red_flags,
                "strong_signals": strong_signals,
                "jd_alignment": jd_alignment,
            },
        )

    def _extract_must_have_from_prior(self, prior_results: list[dict]) -> list[str]:
        for result in prior_results:
            if result.get("agent_name") == "jd_deconstructor":
                data = result.get("data") or {}
                return data.get("must_have", [])
        return []

    def _parse_llm_response(self, raw: str) -> dict | None:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"[{self.agent_name}] JSON parse failed: {e}")
            return None
