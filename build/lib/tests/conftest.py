import pytest
import os
import json
import sys
from unittest.mock import Mock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add src to the Python path to make our custom sha3 module accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import after sys.path changes
from hdwallet import HDWallet

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
                "derivation_path": "m/44'/0'/0'/0/0"
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
    redis_mock = mocker.AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
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
        'mnemonic': 'test mnemonic'
    }
    
    # Create mock for HDWallet class
    mock_instance = mocker.Mock()
    mock_instance.dumps.return_value = wallet_data
    mock_instance.from_mnemonic.return_value = None
    mock_instance.from_path.return_value = None
    
    # Patch the HDWallet constructor and return our mock
    mocker.patch('hdwallet.HDWallet', return_value=mock_instance)
    
    return mock_instance

# Remove redundant fixture to avoid conflicts
@pytest.fixture(autouse=False)
def mock_hdwallet_globally(mocker):
    pass
