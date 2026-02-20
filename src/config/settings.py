import json
from pathlib import Path

from pydantic import Field, field_validator
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
    db_auth_plugin: str | None = None
    smtp_host: str = Field("smtp.gmail.com", validation_alias="SMTP_HOST")
    smtp_port: int = Field(587, validation_alias="SMTP_PORT")
    smtp_user: str | None = Field(None, validation_alias="SMTP_USER")
    smtp_pass: str | None = Field(None, validation_alias="SMTP_PASS")
    smtp_from: str | None = Field(None, validation_alias="SMTP_FROM")
    smtp_use_tls: bool = Field(True, validation_alias="SMTP_USE_TLS")
    reset_base_url: str | None = Field(None, validation_alias="RESET_BASE_URL")
    redis_url: str | None = Field(None, validation_alias="REDIS_URL")
    redis_cluster_mode: int = Field(0, validation_alias="REDIS_CLUSTER_MODE")
    redis_host: str = Field("127.0.0.1", validation_alias="REDIS_HOST")
    redis_port: int = Field(6379, validation_alias="REDIS_PORT")
    redis_pwd: str | None = Field(None, validation_alias="REDIS_PWD")
    redis_queue: str = Field("document_pipeline", validation_alias="REDIS_QUEUE")
    redis_stream: str = Field("document_pipeline", validation_alias="REDIS_STREAM")
    redis_stream_maxlen: int | None = Field(
        None, validation_alias="REDIS_STREAM_MAXLEN"
    )

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
    config = {
        "host": settings.db_host,
        "port": settings.db_port,
        "user": settings.db_user,
        "password": settings.db_pwd,
        "db": settings.db_name,
        "charset": "utf8mb4",
    }
    if settings.db_auth_plugin:
        config["auth_plugin"] = settings.db_auth_plugin
    return config
