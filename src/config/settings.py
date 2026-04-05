import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.resources.crypto_env import decrypt_secret

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        enable_decoding=False,
    )

    app_name: str = "kl_api"
    env: str = "local"
    expose_internal_error_detail: bool = Field(
        False,
        validation_alias="EXPOSE_INTERNAL_ERROR_DETAIL",
    )
    cors_origins: list[str] = ["http://localhost:5173","http://127.0.0.1:5173"]
    log_level: str = "INFO"
    db_host: str = Field("127.0.0.1", validation_alias="DB_HOST")
    db_port: int = Field(3306, validation_alias="DB_PORT")
    db_user: str = Field("root", validation_alias="DB_USER")
    db_pwd: str = Field("", validation_alias="DB_PWD")
    db_name: str = Field("kl_api", validation_alias="DATABASE")
    db_auth_plugin: str | None = None
    db_connect_timeout: float = Field(5.0, validation_alias="DB_CONNECT_TIMEOUT")
    db_read_timeout: float = Field(10.0, validation_alias="DB_READ_TIMEOUT")
    db_write_timeout: float = Field(10.0, validation_alias="DB_WRITE_TIMEOUT")
    db_pool_acquire_timeout: float = Field(
        5.0, validation_alias="DB_POOL_ACQUIRE_TIMEOUT"
    )
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
    worker_batch_stream: str | None = Field(
        None, validation_alias="KL_WORKER_BATCH_STREAM"
    )
    worker_chat_stream: str | None = Field(
        None, validation_alias="KL_WORKER_CHAT_STREAM"
    )
    redis_pipeline_status_stream: str | None = Field(
        None, validation_alias="KL_WORKER_REDIS_TASK_STATUS_STREAM"
    )
    # Backward-compat for deployments using KL_WORKER_REDIS_STREAM.
    redis_stream_key: str | None = Field(None, validation_alias="KL_WORKER_REDIS_STREAM")
    redis_stream_maxlen: int | None = Field(
        None, validation_alias="REDIS_STREAM_MAXLEN"
    )
    rag_chat_top_k: int = Field(5, validation_alias="RAG_CHAT_TOP_K")
    jwt_secret_key: str = Field("change-me", validation_alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field("HS256", validation_alias="JWT_ALGORITHM")
    jwt_expire_min: int = Field(60, validation_alias="JWT_EXPIRE_MIN")
    auth_enabled: bool = Field(True, validation_alias="AUTH_ENABLED")
    storage_provider: str = Field("minio", validation_alias="STORAGE_PROVIDER")
    storage_endpoint: str = Field("", validation_alias="STORAGE_ENDPOINT")
    storage_use_ssl: bool = Field(False, validation_alias="STORAGE_USE_SSL")
    storage_path_style: bool = Field(True, validation_alias="STORAGE_PATH_STYLE")
    storage_region: str | None = Field(None, validation_alias="STORAGE_REGION")
    storage_bucket_name: str = Field("", validation_alias="STORAGE_BUCKET_NAME")
    storage_base_prefix: str = Field("", validation_alias="STORAGE_BASE_PREFIX")
    storage_access_key: str | None = Field(
        None, validation_alias="STORAGE_ACCESS_KEY"
    )
    storage_secret_key: str | None = Field(
        None, validation_alias="STORAGE_SECRET_KEY"
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

    def model_post_init(self, __context) -> None:
        legacy_stream = self.redis_stream_key or self.redis_stream
        if not self.worker_batch_stream:
            self.worker_batch_stream = legacy_stream
        if not self.worker_chat_stream:
            self.worker_chat_stream = legacy_stream
        # If REDIS_STREAM_KEY is provided and REDIS_STREAM is left at default,
        # prefer the key for stream name to avoid misconfiguration.
        if self.redis_stream_key and self.redis_stream == "document_pipeline":
            self.redis_stream = self.redis_stream_key


class WorkerMessageType:
    INGESTION = "ingestion"
    QA_GENERATION = "qa_generation"
    RAG_CHAT = "rag-chat"
    DOC_CHAT = "doc-chat"
    CHAT_CANCEL = "chat-cancel"
    CABINET_COLLECTION_DELETE = "cabinet_collection_delete"


@dataclass(frozen=True)
class WorkerStreamRoute:
    route_name: str
    env_var: str
    stream_name: str | None
    message_types: frozenset[str]


class WorkerStreamRouter:
    def __init__(self, app_settings: "Settings"):
        self._routes = (
            WorkerStreamRoute(
                route_name="batch",
                env_var="KL_WORKER_BATCH_STREAM",
                stream_name=app_settings.worker_batch_stream,
                message_types=frozenset(
                    {
                        WorkerMessageType.INGESTION,
                        WorkerMessageType.QA_GENERATION,
                        WorkerMessageType.CABINET_COLLECTION_DELETE,
                    }
                ),
            ),
            WorkerStreamRoute(
                route_name="chat",
                env_var="KL_WORKER_CHAT_STREAM",
                stream_name=app_settings.worker_chat_stream,
                message_types=frozenset(
                    {
                        WorkerMessageType.DOC_CHAT,
                        WorkerMessageType.RAG_CHAT,
                        WorkerMessageType.CHAT_CANCEL,
                    }
                ),
            ),
        )
        self._routes_by_message_type = {
            message_type: route
            for route in self._routes
            for message_type in route.message_types
        }

    @property
    def routes(self) -> tuple[WorkerStreamRoute, ...]:
        return self._routes

    def route_for_type(self, message_type: str) -> WorkerStreamRoute:
        route = self._routes_by_message_type.get(message_type)
        if route is None:
            raise KeyError(f"Unsupported worker message type: {message_type}")
        return route

    def stream_for_type(self, message_type: str) -> str | None:
        return self.route_for_type(message_type).stream_name


settings = Settings()
worker_stream_router = WorkerStreamRouter(settings)


def get_jwt_secret_key() -> str:
    secret = settings.jwt_secret_key
    if isinstance(secret, str) and secret.startswith("ENC"):
        return decrypt_secret(secret)
    return secret


def build_mysql_aiomysql_config() -> dict[str, object]:
    password = settings.db_pwd
    if isinstance(password, str) and password.startswith("ENC"):
        password = decrypt_secret(password)
    config = {
        "host": settings.db_host,
        "port": int(settings.db_port),
        "user": settings.db_user,
        "password": password,
        "db": settings.db_name,
        "charset": "utf8mb4",
        "connect_timeout": settings.db_connect_timeout,
        "read_timeout": settings.db_read_timeout,
        "write_timeout": settings.db_write_timeout,
    }
    if settings.db_auth_plugin:
        config["auth_plugin"] = settings.db_auth_plugin
    return config
