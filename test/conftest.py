import os
import pytest
from unittest.mock import patch
from app.core.settings import get_settings

TEST_ENV_VARS = {
    "APP_ENV": "development",
    "DEBUG": "true",
    "LLM_PROVIDER": "openai",
    "OPENAI_API_KEY": "sk-test-fake-key-not-real",
    "ANTHROPIC_API_KEY": "sk-ant-test-fake-key-not-real",
    "CONFIDENCE_THRESHOLD": "0.5",
    "LOG_LEVEL": "DEBUG",
}


@pytest.fixture(autouse=True, scope="session")
def set_test_environment():
    with patch.dict(os.environ, TEST_ENV_VARS, clear=False):
        get_settings.cache_clear()  # Force settings to reload with test env
        yield
        get_settings.cache_clear()  # Clean up after session


@pytest.fixture
def sample_hiring_payload() -> dict:
    return {
        "candidate_name": "Alice Example",
        "job_title": "Senior Backend Engineer",
        "resume_text": "10 years of Python experience...",
    }


@pytest.fixture
def sample_procurement_payload() -> dict:
    return {
        "item": "Laptop",
        "quantity": 50,
        "department": "Engineering",
        "estimated_budget": 75000,
    }