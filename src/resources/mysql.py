from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiomysql

from src.config import build_mysql_aiomysql_config


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
            self._pool = await aiomysql.create_pool(
                minsize=self._pool_minsize,
                maxsize=self._pool_maxsize,
                pool_recycle=self._pool_recycle,
                autocommit=False,
                **self._conn_kwargs,
            )
        return self._pool

    @asynccontextmanager
    async def async_conn(self) -> AsyncIterator[aiomysql.Connection]:
        pool = await self._get_pool()
        conn = await pool.acquire()
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


async def async_get_db() -> AsyncIterator[aiomysql.Connection]:
    async with _mysql_resource.async_conn() as conn:
        yield conn


async def close_mysql() -> None:
    await _mysql_resource.close()
