from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.resources.mysql import close_mysql
from src.resources.redis import close_redis, get_redis_client
from src.routers import api_router

_LOGGING_CONFIGURED = False


def _configure_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    level_name = (settings.log_level or "INFO").upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _LOGGING_CONFIGURED = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup/shutdown hooks live here.
    try:
        _configure_logging()
        try:
            await get_redis_client()
        except RuntimeError:
            pass
        yield
    finally:
        await close_mysql()
        await close_redis()


app = FastAPI(lifespan=lifespan, title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Request-Id"] = request.headers.get("X-Request-Id", "")
    return response


app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=os.getenv("UVICORN_HOST", "0.0.0.0"),
        port=int(os.getenv("UVICORN_PORT", "8000")),
        reload=os.getenv("UVICORN_RELOAD", "false").lower() == "true",
        timeout_graceful_shutdown=int(
            os.getenv("UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN", "3")
        ),
    )
