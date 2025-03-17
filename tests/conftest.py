import pytest
import os
import json
import sys
from unittest.mock import Mock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add src to the Python path to make our custom sha3 module accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import sha3 replacement first
from src.sha3 import keccak_256
import sys
sys.modules['sha3'] = sys.modules['src.sha3']

from hdwallet import HDWallet
from hdwallet.symbols import BTC

@pytest.fixture(autouse=True, scope="session")
def setup_test_env(tmp_path_factory):
    test_dir = tmp_path_factory.mktemp("test_btcminer")
    
    os.environ['CONFIG_FILE'] = str(test_dir / 'config.json')
    
    test_config = {
        "settings": {
            "bruteforcer": {
                "strength": 128,
                "language": "english",
                "passphrase": "test"
            },
            "checker": {
                "filename": str(test_dir / "check.txt")
            },
            "general": {
                "failed": str(test_dir / "failed.txt"),
                "success": str(test_dir / "success.txt"),
                "addresstype": "p2pkh",
                "api": {
                    "api_url": "https://test.api.com",
                    "api_get_data": "data",
                    "api_get_balance": "balance",
                    "api_get_received": "received"
                },
                "derivation_path": "m/44'/0'/0'/0/0",
                "redis_url": "redis://localhost",
                "metrics_port": 9090,
                "proxies": ["http://proxy1.example.com", "http://proxy2.example.com"],
                "proxy_score_threshold": 5,
                "proxy_timeout": 10
            }
        }
    }
    
    with open(os.environ['CONFIG_FILE'], 'w') as f:
        json.dump(test_config, f)
    
    for file in ["check.txt", "failed.txt", "success.txt"]:
        open(test_dir / file, 'a').close()
    
    yield test_dir
    
    os.environ.pop('CONFIG_FILE', None)

@pytest.fixture(autouse=True)
def mock_redis_service(mocker):
    """Create a fully mocked Redis service with working async context manager for pipelines"""
    # Set up the pipeline mock
    pipeline_mock = mocker.AsyncMock()
    pipeline_mock.get.return_value = None
    pipeline_mock.ttl.return_value = 0
    pipeline_mock.execute.return_value = [None, 0]
    
    # Set up the context manager correctly
    pipeline_context = mocker.MagicMock()
    # Make __aenter__ a coroutine that returns the pipeline_mock
    pipeline_context.__aenter__ = mocker.AsyncMock(return_value=pipeline_mock)
    pipeline_context.__aexit__ = mocker.AsyncMock(return_value=None)
    
    # Create the Redis mock
    redis_mock = mocker.AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    
    # Make pipeline() return the context manager directly (not a coroutine)
    redis_mock.pipeline = mocker.MagicMock(return_value=pipeline_context)
    
    return redis_mock

@pytest.fixture
def mock_redis(mocker):
    redis_mock = mocker.MagicMock()
    mocker.patch('redis.asyncio.from_url', return_value=redis_mock)
    return redis_mock

@pytest.fixture(autouse=True)
def mock_prometheus(mocker):
    mocker.patch('prometheus_client.start_http_server')
    return mocker

@pytest.fixture(autouse=True)
def mock_hdwallet(mocker):
    # Create wallet data with the expected structure
    wallet_data = {
        'addresses': {'p2pkh': '1test'},
        'wif': 'testwif',
        'entropy': 'testentropy',
        'private_key': 'testkey',
        'mnemonic': 'abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about'
    }
    
    # Create mock for HDWallet class
    mock_instance = mocker.Mock()
    mock_instance.dumps.return_value = wallet_data
    mock_instance.from_mnemonic.return_value = None
    mock_instance.from_path.return_value = None
    
    # Patch the HDWallet constructor and return our mock
    mocker.patch('hdwallet.HDWallet', return_value=mock_instance)
    
    return mock_instance
