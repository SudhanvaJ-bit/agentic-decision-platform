from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  
        extra="ignore",         
    )

    app_name: str = Field(default="Enterprise Agent Platform")
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)

    llm_provider: Literal["openai", "anthropic", "azure_openai"] = Field(default="openai")
    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")

    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def active_llm_api_key(self) -> str:
        if self.llm_provider == "openai":
            return self.openai_api_key
        elif self.llm_provider == "anthropic":
            return self.anthropic_api_key
        return ""

    @field_validator("confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {v}")
        return v


@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()