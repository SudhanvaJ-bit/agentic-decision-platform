import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from app.core.settings import settings

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,   # Low temp = deterministic, consistent outputs
        max_tokens: int = 1500,
    ) -> str:

        ...

class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is not installed. "
                "Run: pip install openai"
            )

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set in your .env file. "
                "Add it before using the real LLM provider."
            )

        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o-mini"   # Cost-efficient, strong reasoning
        logger.info(f"OpenAIProvider initialized with model: {self.model}")

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        logger.debug(f"OpenAI request | model={self.model} | temp={temperature}")

        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content or ""
        logger.debug(f"OpenAI response received | chars={len(content)}")
        return content

class MockLLMProvider(BaseLLMProvider):
    def __init__(self, response: str | None = None):
        self._response = response
        self._call_count = 0
        logger.info("MockLLMProvider initialized — no real API calls will be made.")

    def set_response(self, response: str) -> None:
        self._response = response

    @property
    def call_count(self) -> int:
        return self._call_count

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        self._call_count += 1
        logger.debug(f"MockLLMProvider.complete() called (call #{self._call_count})")

        if self._response is None:
            # Return a safe, parseable default so agents don't crash
            # when no custom response is set
            return json.dumps({
                "summary": "Mock LLM response",
                "reasoning": ["This is a mock response for testing."],
                "confidence": 0.9,
                "data": {}
            })

        return self._response

def get_llm_provider() -> BaseLLMProvider:
    provider = settings.llm_provider
    has_openai_key = bool(settings.openai_api_key and
                          not settings.openai_api_key.startswith("sk-placeholder"))
    has_anthropic_key = bool(settings.anthropic_api_key and
                             not settings.anthropic_api_key.startswith("sk-ant-placeholder"))

    if provider == "openai" and has_openai_key:
        logger.info("LLM Factory: returning OpenAIProvider")
        return OpenAIProvider()

    # Fall through to mock if no valid key is present
    logger.warning(
        "LLM Factory: No valid API key found. "
        "Returning MockLLMProvider. "
        "Set a real API key in .env to use live LLM calls."
    )
    return MockLLMProvider()