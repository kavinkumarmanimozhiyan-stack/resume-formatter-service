import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")

IS_VERCEL = bool(os.getenv("VERCEL"))

class Settings(BaseSettings):
    GOOGLE_API_KEY: Optional[str] = None
    # PORT: int = 8000
    MAX_FILE_SIZE_MB: int = 10
    # Caps how many full resume-generation pipelines (Gemini calls +
    # rasterization + Chromium + LibreOffice) run at once. Default 1 —
    # tuned for a 1 GB instance where two full pipelines don't fit
    # simultaneously. Excess requests queue on the semaphore rather than
    # piling up memory in parallel. Raise only after benchmarking headroom.
    MAX_CONCURRENT_PIPELINES: int = 1
    # Comma-separated frontend origins permitted to call this API from a browser.
    # Add the deployed frontend URL here (for example, https://app.example.com).
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:3001,http://localhost:5173"
    # LLM_SETTINGS_DB_PATH: str = "data/app.db"
    LLM_SETTINGS_DB_PATH: str = (
    "/tmp/data/llm_settings.db"
    if IS_VERCEL
    else "data/llm_settings.db"
)
    LLM_SETTINGS_ENCRYPTION_KEY: Optional[str] = None

    # Cloudinary settings
    CLOUDINARY_CLOUD_NAME: Optional[str] = None
    CLOUDINARY_API_KEY: Optional[str] = None
    CLOUDINARY_API_SECRET: Optional[str] = None

    model_config = SettingsConfigDict(env_file=_ENV_PATH, extra="ignore")


settings = Settings()
