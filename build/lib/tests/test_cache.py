import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.cache import CacheService

@pytest.mark.asyncio
async def test_cache_operations(mock_redis_service):
    # Create cache service instance
    cache = CacheService("redis://localhost")
    # Manually assign the mock redis to the instance
    cache.redis = mock_redis_service
    
    address = "test_address"
    balance = (1.0, 2.0)
    
    # Test cache miss
    mock_redis_service.get.return_value = None
    result = await cache.get_balance(address)
    assert result is None
    
    # Test cache set
    mock_redis_service.set.return_value = True
    await cache.set_balance(address, balance[0], balance[1])
    assert mock_redis_service.set.called
    
    # Test cache hit
    mock_redis_service.get.return_value = f"{balance[0]}:{balance[1]}".encode()
    result = await cache.get_balance(address)
    assert result == balance
