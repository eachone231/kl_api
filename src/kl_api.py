from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.resources.auth import verify_access_token
from src.resources.mysql import close_mysql
from src.resources.redis import close_redis, get_redis_client
from src.routers import api_router

_LOGGING_CONFIGURED = False
_PUBLIC_PATHS = {
    "/",
    "/health",
    "/status",
    "/api/login",
    "/api/user",
    "/api/user/reset",
    "/api/user/reset/confirm",
    "/api/pipeline/status",
}


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
    if request.method == "OPTIONS":
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.headers.get("X-Request-Id", "")
        return response

    path = request.url.path
    if settings.auth_enabled and path.startswith("/api") and path not in _PUBLIC_PATHS:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing Bearer token"},
            )
        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing Bearer token"},
            )
        try:
            request.state.auth_user = verify_access_token(token)
        except HTTPException:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired token"},
            )

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
