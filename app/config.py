from functools import lru_cache

from pydantic import Field
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception:  # noqa: BLE001
    from pydantic import BaseSettings  # type: ignore[assignment]

    SettingsConfigDict = None


class Settings(BaseSettings):
    openrouter_api_key: str = Field(default="")
    openrouter_model: str = Field(default="mistralai/mistral-7b-instruct")
    openrouter_url: str = Field(default="https://openrouter.ai/api/v1/chat/completions")
    openrouter_http_referer: str = Field(default="http://localhost:8000")
    openrouter_app_title: str = Field(default="Dynamic AI Mock Server")

    generation_temperature: float = Field(default=0.3)
    generation_max_tokens: int = Field(default=512)
    generation_timeout_seconds: int = Field(default=45)
    max_retry_attempts: int = Field(default=3)
    max_multi_response: int = Field(default=10)

    semantic_model_name: str = Field(default="all-MiniLM-L6-v2")
    semantic_similarity_threshold: float = Field(default=0.6)
    semantic_fail_open: bool = Field(default=True)

    context_history_limit: int = Field(default=200)

    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
