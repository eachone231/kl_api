from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.resources.mysql import close_mysql
from src.resources.redis import close_redis, get_redis_client
from src.routers import api_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup/shutdown hooks live here.
    try:
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
