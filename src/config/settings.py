import json
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        enable_decoding=False,
    )

    app_name: str = "kl_api"
    env: str = "local"
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"
    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "root"
    db_pwd: str = ""
    db_name: str = "kl_api"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, list):
            return value
        if not isinstance(value, str):
            return ["*"]
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in raw.split(",") if item.strip()]


settings = Settings()


def build_mysql_aiomysql_config() -> dict[str, object]:
    return {
        "host": settings.db_host,
        "port": settings.db_port,
        "user": settings.db_user,
        "password": settings.db_pwd,
        "db": settings.db_name,
        "charset": "utf8mb4",
    }
