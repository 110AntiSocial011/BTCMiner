import pytest
import sys
import os
import pickle

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
    
    # Test cache miss - no need to await pipeline() since it's now a MagicMock
    pipeline_context = mock_redis_service.pipeline()
    pipeline_mock = await pipeline_context.__aenter__()
    pipeline_mock.execute.return_value = [None, 0]  # Empty result for get and ttl
    
    result = await cache.get_balance(address)
    assert result is None
    
    # Test cache set
    mock_redis_service.set.return_value = True
    success = await cache.set_balance(address, balance[0], balance[1])
    assert success is True
    
    # Test cache hit - reconfigure the mock for the hit case
    pickled_data = pickle.dumps(balance)
    pipeline_mock.execute.return_value = [pickled_data, 100]  # Pickled data and TTL
    
    result = await cache.get_balance(address)
    assert result == balance
