"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized settings for the CREDORA backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "CREDORA"
    APP_TAGLINE: str = "TRUST IN EVERY TRANSACTION."
    APP_ENV: str = "development"
    DEBUG: bool = False

    DATABASE_URL: str = (
        "postgresql+psycopg2://credora:credora_dev_password@localhost:5432/credora_db"
    )

    JWT_SECRET_KEY: str = "credora-dev-secret-change-in-production-9f8a7b6c5d4e"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    ML_MODELS_DIR: str = "../ml_models"
    DATASETS_DIR: str = "../datasets"

    MAX_CSV_UPLOAD_MB: int = 25
    DEFAULT_RISK_MEDIUM_THRESHOLD: float = 30.0
    DEFAULT_RISK_HIGH_THRESHOLD: float = 70.0

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def ml_models_path(self) -> Path:
        p = Path(self.ML_MODELS_DIR)
        if not p.is_absolute():
            p = (Path(__file__).resolve().parents[2] / p).resolve()
        return p

    @property
    def datasets_path(self) -> Path:
        p = Path(self.DATASETS_DIR)
        if not p.is_absolute():
            p = (Path(__file__).resolve().parents[2] / p).resolve()
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
