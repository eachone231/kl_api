class RedisClient:
    def __init__(self, url: str):
        self._url = url
        self._connected = False

    async def connect(self) -> None:
        # TODO: initialize redis client here.
        self._connected = True

    async def close(self) -> None:
        # TODO: close redis client here.
        self._connected = False

    @property
    def url(self) -> str:
        return self._url

    @property
    def connected(self) -> bool:
        return self._connected
