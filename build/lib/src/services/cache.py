import redis.asyncio as redis
from typing import Optional, Tuple

class CacheService:
    def __init__(self, redis_url: str):
        self.redis = None
        self.redis_url = redis_url

    async def initialize(self):
        try:
            self.redis = await redis.from_url(self.redis_url)
        except Exception as e:
            self.redis = None
            return False

    async def get_balance(self, address: str) -> Optional[Tuple[float, float]]:
        if not self.redis:
            return None
        try:
            cached = await self.redis.get(address)
            if cached:
                balance, received = cached.decode().split(':')
                return float(balance), float(received)
        except Exception:
            pass
        return None

    async def set_balance(self, address: str, balance: float, received: float) -> bool:
        if not self.redis:
            return False
        try:
            value = f"{balance}:{received}".encode()
            return await self.redis.set(address, value, ex=3600)  # 1 hour expiration
        except Exception:
            return False
