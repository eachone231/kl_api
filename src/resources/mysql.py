from __future__ import annotations

from contextlib import asynccontextmanager
import inspect
import asyncio
from typing import AsyncIterator

import aiomysql
import pymysql

from src.config import build_mysql_aiomysql_config
from src.config import settings


class MySQLResource:
    def __init__(
        self,
        conn_kwargs: dict[str, object],
        pool_minsize: int = 1,
        pool_maxsize: int = 10,
        pool_recycle: int = 3600,
    ):
        self._conn_kwargs = conn_kwargs
        self._pool_minsize = pool_minsize
        self._pool_maxsize = pool_maxsize
        self._pool_recycle = pool_recycle
        self._pool: aiomysql.Pool | None = None

    async def _get_pool(self) -> aiomysql.Pool:
        if self._pool is None:
            conn_kwargs = _filter_conn_kwargs(self._conn_kwargs)
            self._pool = await aiomysql.create_pool(
                minsize=self._pool_minsize,
                maxsize=self._pool_maxsize,
                pool_recycle=self._pool_recycle,
                autocommit=False,
                **conn_kwargs,
            )
        return self._pool

    @asynccontextmanager
    async def async_conn(self) -> AsyncIterator[aiomysql.Connection]:
        pool = await self._get_pool()
        conn = await asyncio.wait_for(
            pool.acquire(),
            timeout=settings.db_pool_acquire_timeout,
        )
        try:
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            pool.release(conn)

    async def ping(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT 1")

    async def close(self) -> None:
        if self._pool is None:
            return
        self._pool.close()
        await self._pool.wait_closed()
        self._pool = None

_mysql_resource = MySQLResource(
    conn_kwargs=build_mysql_aiomysql_config(),
)


def _strict_signature(candidate: object) -> inspect.Signature | None:
    try:
        sig = inspect.signature(candidate)
    except (TypeError, ValueError):
        return None
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return None
    return sig


def _filter_conn_kwargs(conn_kwargs: dict[str, object]) -> dict[str, object]:
    aiomysql_sig = _strict_signature(aiomysql.connect)
    if aiomysql_sig is not None:
        allowed = set(aiomysql_sig.parameters.keys())
        allowed.discard("self")
        return {key: value for key, value in conn_kwargs.items() if key in allowed}

    for candidate in (
        getattr(pymysql, "connect", None),
        getattr(pymysql.connections, "Connection", None),
    ):
        sig = _strict_signature(candidate)
        if sig is None:
            continue
        allowed = set(sig.parameters.keys())
        allowed.discard("self")
        return {key: value for key, value in conn_kwargs.items() if key in allowed}

    return conn_kwargs


async def async_get_db() -> AsyncIterator[aiomysql.Connection]:
    async with _mysql_resource.async_conn() as conn:
        yield conn


async def close_mysql() -> None:
    await _mysql_resource.close()
