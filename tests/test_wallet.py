import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add src to the Python path to make our custom sha3 module accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.models.wallet import BTCWallet
from src.models.config import Config
from hdwallet import HDWallet
from hdwallet.symbols import BTC

@pytest.fixture
def clean_wallet_cache():
    """Reset the wallet cache between tests"""
    BTCWallet._wallet_cache = {}
    yield
    BTCWallet._wallet_cache = {}

@pytest.fixture
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
    mocker.patch('src.models.wallet.HDWallet', return_value=mock_instance)
    
    return mock_instance

def test_wallet_creation(setup_test_env, mock_hdwallet, clean_wallet_cache):
    config = Config.load()
    mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
    wallet = BTCWallet.create_from_mnemonic(mnemonic, config)
    
    # Using the exact mock values to compare
    assert wallet.seed == mnemonic  # Seed is the original mnemonic
    assert wallet.private_key == "testkey"
    assert wallet.wif == "testwif"
    assert wallet.addresses == {"p2pkh": "1test"}

def test_check_balance(mocker, setup_test_env, mock_hdwallet, clean_wallet_cache):
    config = Config.load()
    mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
    wallet = BTCWallet.create_from_mnemonic(mnemonic, config)
    
    # Mock requests.get
    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "data": {
            "balance": "1.0",
            "received": "2.0"
        }
    }
    mocker.patch('requests.get', return_value=mock_response)
    
    # Clear cachetools cache to ensure fresh test
    wallet.check_balance.cache_clear()
    
    balance, received = wallet.check_balance(list(wallet.addresses.values())[0])
    assert balance == 1.0
    assert received == 2.0
