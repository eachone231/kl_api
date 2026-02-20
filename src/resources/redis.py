import asyncio
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from src.config import settings


@dataclass(frozen=True)
class RedisConfig:
    host: str
    port: int
    password: str | None
    db: int


class RedisClient:
    def __init__(self, url: str, queue: str, stream: str, stream_maxlen: int | None):
        if not url:
            raise ValueError("Redis URL is required")
        self._url = url
        self._queue = queue
        self._stream = stream
        self._stream_maxlen = stream_maxlen
        self._config = self._parse_url(url)
        self._connected = False
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _parse_url(url: str) -> RedisConfig:
        parsed = urlparse(url)
        if parsed.scheme not in ("redis", "rediss"):
            raise ValueError("Redis URL must start with redis:// or rediss://")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 6379
        password = parsed.password
        db = 0
        if parsed.path and parsed.path != "/":
            try:
                db = int(parsed.path.lstrip("/"))
            except ValueError as exc:
                raise ValueError("Redis URL DB must be an integer") from exc
        return RedisConfig(host=host, port=port, password=password, db=db)

    async def connect(self) -> None:
        if self._connected:
            return
        reader, writer = await asyncio.open_connection(
            self._config.host, self._config.port
        )
        self._reader = reader
        self._writer = writer
        if self._config.password:
            await self._execute_locked("AUTH", self._config.password)
        if self._config.db:
            await self._execute_locked("SELECT", str(self._config.db))
        self._connected = True

    async def close(self) -> None:
        if not self._connected:
            return
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        self._connected = False

    @property
    def url(self) -> str:
        return self._url

    @property
    def queue(self) -> str:
        return self._queue

    @property
    def stream(self) -> str:
        return self._stream

    @property
    def stream_maxlen(self) -> int | None:
        return self._stream_maxlen

    @property
    def connected(self) -> bool:
        return self._connected

    async def lpush(self, key: str, value: str) -> int:
        result = await self.execute("LPUSH", key, value)
        if not isinstance(result, int):
            raise RuntimeError("Unexpected Redis LPUSH response")
        return result

    async def xadd(
        self,
        stream: str,
        fields: dict[str, str],
        maxlen: int | None = None,
    ) -> str:
        if not fields:
            raise ValueError("XADD fields are required")
        args: list[str] = ["XADD", stream]
        if maxlen is not None:
            args.extend(["MAXLEN", "~", str(maxlen)])
        args.append("*")
        for key, value in fields.items():
            args.append(str(key))
            args.append(str(value))
        result = await self.execute(*args)
        if not isinstance(result, str):
            raise RuntimeError("Unexpected Redis XADD response")
        return result

    async def execute(self, *args: str) -> Any:
        if not self._connected:
            await self.connect()
        return await self._execute_locked(*args)

    async def _execute_locked(self, *args: str) -> Any:
        async with self._lock:
            if not self._reader or not self._writer:
                raise RuntimeError("Redis connection is not available")
            self._writer.write(self._encode_command(*args))
            await self._writer.drain()
            return await self._read_response()

    @staticmethod
    def _encode_command(*parts: str) -> bytes:
        encoded = [f"*{len(parts)}\r\n".encode("utf-8")]
        for part in parts:
            if not isinstance(part, str):
                part = str(part)
            data = part.encode("utf-8")
            encoded.append(f"${len(data)}\r\n".encode("utf-8"))
            encoded.append(data + b"\r\n")
        return b"".join(encoded)

    async def _read_response(self) -> Any:
        if not self._reader:
            raise RuntimeError("Redis connection is not available")
        line = await self._reader.readline()
        if not line:
            raise RuntimeError("Redis connection closed")
        prefix = line[:1]
        payload = line[1:].strip()
        if prefix == b"+":
            return payload.decode("utf-8")
        if prefix == b"-":
            raise RuntimeError(payload.decode("utf-8"))
        if prefix == b":":
            return int(payload)
        if prefix == b"$":
            length = int(payload)
            if length == -1:
                return None
            data = await self._reader.readexactly(length + 2)
            return data[:-2].decode("utf-8")
        if prefix == b"*":
            count = int(payload)
            if count == -1:
                return None
            return [await self._read_response() for _ in range(count)]
        raise RuntimeError("Unexpected Redis response")


def _build_redis_url() -> str | None:
    from urllib.parse import quote

    if settings.redis_url:
        return settings.redis_url
    host = settings.redis_host
    port = settings.redis_port
    password = settings.redis_pwd
    if not host or not port:
        return None
    auth = ""
    if password:
        auth = f":{quote(password, safe='')}@"
    return f"redis://{auth}{host}:{port}/0"


_redis_client: RedisClient | None = None
_redis_url = _build_redis_url()
if _redis_url:
    _redis_client = RedisClient(
        _redis_url,
        settings.redis_queue,
        settings.redis_stream,
        settings.redis_stream_maxlen,
    )


async def get_redis_client() -> RedisClient:
    if _redis_client is None:
        raise RuntimeError("REDIS_URL is not configured")
    await _redis_client.connect()
    return _redis_client


async def close_redis() -> None:
    if _redis_client is None:
        return
    await _redis_client.close()
