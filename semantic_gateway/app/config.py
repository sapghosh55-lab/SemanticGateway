from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for SemanticRouter gateway loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "../../.env", "../../../.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Gateway Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ROUTES_FILE: str = "routes.yaml"

    # Embedding & Classification Settings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD: float = 0.45
    HF_TOKEN: str = ""
    HUGGINGFACE_API_KEY: str = ""

    # Fast Lane LLM Tier (Groq Llama 3.1 8B Fast Model)
    FAST_MODEL_NAME: str = "llama-3.1-8b-instant"
    FAST_MODEL_BASE_URL: str = "https://api.groq.com/openai/v1"
    FAST_MODEL_API_KEY: str = ""
    FAST_MODEL_COST_PER_1K_INPUT: float = 0.0001
    FAST_MODEL_COST_PER_1K_OUTPUT: float = 0.0002

    # Deep Lane LLM Tier Settings (Groq Llama 3.3 70B Frontier Model)
    DEEP_MODEL_NAME: str = "llama-3.3-70b-versatile"
    DEEP_MODEL_BASE_URL: str = "https://api.groq.com/openai/v1"
    DEEP_BASE_URL: str = "https://api.groq.com/openai/v1"
    DEEP_MODEL_API_KEY: str = ""

    FRONTIER_MODEL_NAME: str = "llama-3.3-70b-versatile"
    FRONTIER_MODEL_BASE_URL: str = "https://api.groq.com/openai/v1"
    FRONTIER_MODEL_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # Google Gemini OpenAI-compatible Settings
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_API_KEY: str = ""

    FRONTIER_MODEL_COST_PER_1K_INPUT: float = 0.0059
    FRONTIER_MODEL_COST_PER_1K_OUTPUT: float = 0.0079


def get_settings() -> Settings:
    return Settings()
