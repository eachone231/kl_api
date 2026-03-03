from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mysql import MySQLResource
    from .redis import RedisClient

__all__ = ["MySQLResource", "RedisClient"]


def __getattr__(name: str) -> Any:
    if name == "MySQLResource":
        from .mysql import MySQLResource

        return MySQLResource
    if name == "RedisClient":
        from .redis import RedisClient

        return RedisClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
