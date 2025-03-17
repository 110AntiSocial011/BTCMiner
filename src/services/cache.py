import redis.asyncio as redis
from typing import Optional, Tuple, Dict, Any
import pickle
import hashlib
import asyncio
import logging
from functools import wraps

class CacheService:
    def __init__(self, redis_url: str):
        self.redis = None
        self.redis_url = redis_url
        self.local_cache = {}  # Memory cache for faster access
        self.lock = asyncio.Lock()

    async def initialize(self):
        try:
            # Use connection pool for better performance
            self.redis = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # Raw responses for binary data
                socket_timeout=2.0,  # Short timeout
                socket_keepalive=True,  # Keep connections alive
                health_check_interval=30  # Regular health checks
            )
            return True
        except Exception as e:
            logging.error(f"Redis initialization error: {e}")
            self.redis = None
            return False

    async def get_balance(self, address: str) -> Optional[Tuple[float, float]]:
        # Check local cache first (faster than Redis)
        if address in self.local_cache:
            return self.local_cache[address]
            
        if not self.redis:
            return None
            
        try:
            # Get from Redis with pipeline for efficiency
            async with self.redis.pipeline() as pipe:
                await pipe.get(f"balance:{address}")
                await pipe.ttl(f"balance:{address}")
                result, ttl = await pipe.execute()
                
            if result:
                balance_data = pickle.loads(result)
                # Update local cache if TTL is reasonable
                if ttl > 5:  # Only cache if more than 5 seconds TTL left
                    async with self.lock:
                        self.local_cache[address] = balance_data
                return balance_data
        except Exception as e:
            logging.error(f"Redis get error: {e}")
            
        return None

    async def set_balance(self, address: str, balance: float, received: float, ttl: int = 3600) -> bool:
        data = (balance, received)
        
        # Update local cache immediately
        async with self.lock:
            self.local_cache[address] = data
            
        if not self.redis:
            return False
            
        try:
            # Store in Redis with pickle for better serialization
            return await self.redis.set(
                f"balance:{address}",
                pickle.dumps(data),
                ex=ttl
            )
        except Exception as e:
            logging.error(f"Redis set error: {e}")
            return False
            
    # Add cache decorator for other methods
    @staticmethod
    def cache_result(ttl: int = 3600):
        def decorator(func):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                # Generate unique key based on function name and arguments
                key = f"{func.__name__}:{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"
                
                # Try getting from cache
                cached_result = await self.get_value(key)
                if cached_result is not None:
                    return cached_result
                
                # Execute original function
                result = await func(self, *args, **kwargs)
                
                # Cache the result
                await self.set_value(key, result, ttl)
                return result
            return wrapper
        return decorator

    # Add missing methods needed by cache_result decorator
    async def get_value(self, key: str) -> Any:
        """Get a value from the cache by key"""
        if key in self.local_cache:
            return self.local_cache[key]
            
        if not self.redis:
            return None
            
        try:
            result = await self.redis.get(f"value:{key}")
            if result:
                value = pickle.loads(result)
                # Store in local cache
                async with self.lock:
                    self.local_cache[key] = value
                return value
        except Exception as e:
            logging.error(f"Redis get_value error: {e}")
            
        return None

    async def set_value(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set a value in the cache with a TTL"""
        # Update local cache
        async with self.lock:
            self.local_cache[key] = value
            
        if not self.redis:
            return False
            
        try:
            return await self.redis.set(
                f"value:{key}",
                pickle.dumps(value),
                ex=ttl
            )
        except Exception as e:
            logging.error(f"Redis set_value error: {e}")
            return False
