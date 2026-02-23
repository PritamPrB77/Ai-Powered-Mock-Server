from functools import lru_cache

from pydantic import Field
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception:  # noqa: BLE001
    from pydantic import BaseSettings  # type: ignore[assignment]

    SettingsConfigDict = None


class Settings(BaseSettings):
    generation_provider: str = Field(default="ollama")

    openrouter_enabled: bool = Field(default=False)
    openrouter_api_key: str = Field(default="")
    openrouter_model: str = Field(default="openai/gpt-4o-mini")
    openrouter_url: str = Field(default="https://openrouter.ai/api/v1/chat/completions")
    openrouter_http_referer: str = Field(default="http://localhost:8000")
    openrouter_app_title: str = Field(default="Dynamic AI Mock Server")
    openrouter_fallback_enabled: bool = Field(default=False)

    ollama_enabled: bool = Field(default=True)
    ollama_url: str = Field(default="http://localhost:11434/api/generate")
    ollama_model: str = Field(default="qwen2.5:3b")
    ollama_fallback_enabled: bool = Field(default=False)
    ollama_num_predict: int = Field(default=256)
    ollama_top_k: int = Field(default=20)
    ollama_top_p: float = Field(default=0.9)
    ollama_system_prompt: str = Field(
        default=(
            "You are an API response generator for a dynamic mock server. "
            "Return strictly valid JSON only. "
            "Do not output markdown or explanations. "
            "Follow the response schema exactly and do not add extra fields."
        )
    )

    generation_temperature: float = Field(default=0.3)
    generation_max_tokens: int = Field(default=512)
    generation_timeout_seconds: int = Field(default=45)
    max_retry_attempts: int = Field(default=3)
    max_multi_response: int = Field(default=10)

    seq2seq_enabled: bool = Field(default=False)
    seq2seq_model_name: str = Field(default="google/flan-t5-base")
    seq2seq_max_input_tokens: int = Field(default=1024)
    seq2seq_max_new_tokens: int = Field(default=256)
    seq2seq_temperature: float = Field(default=0.2)
    seq2seq_top_p: float = Field(default=0.95)
    seq2seq_num_beams: int = Field(default=1)
    seq2seq_do_sample: bool = Field(default=False)
    seq2seq_warmup_on_startup: bool = Field(default=True)

    semantic_model_name: str = Field(default="all-MiniLM-L6-v2")
    semantic_validation_enabled: bool = Field(default=True)
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
