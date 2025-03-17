import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add src to the Python path to make our custom sha3 module accessible before the real import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.models.wallet import BTCWallet
from src.models.config import Config
from hdwallet import HDWallet
from hdwallet.symbols import BTC

@pytest.fixture
def mock_hdwallet(mocker):
    # Create mock wallet data
    wallet_data = {
        'addresses': {'p2pkh': '1test'},
        'wif': 'testwif',
        'mnemonic': 'abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about',
        'entropy': 'testentropy',
        'private_key': 'testkey'
    }
    
    # Mock HDWallet class - ensure we patch it correctly
    mock_wallet = mocker.Mock()
    mock_wallet.dumps.return_value = wallet_data
    
    # Patch the HDWallet constructor at the correct import path
    mocker.patch('src.models.wallet.HDWallet', return_value=mock_wallet)
    
    # Ensure these methods are mocked
    mock_wallet.from_mnemonic = mocker.Mock()
    mock_wallet.from_path = mocker.Mock()
    
    return mock_wallet

def test_wallet_creation(setup_test_env, mock_hdwallet):
    config = Config.load()
    mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
    wallet = BTCWallet(mnemonic, config)
    
    # Using the exact mock values to compare
    assert wallet.seed == mnemonic  # Seed is the original mnemonic
    assert wallet.private_key == "testkey"
    assert wallet.wif == "testwif"
    assert wallet.addresses == {"p2pkh": "1test"}

def test_check_balance(mocker, setup_test_env, mock_hdwallet):
    config = Config.load()
    mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
    wallet = BTCWallet(mnemonic, config)
    
    # Mock requests.get
    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "data": {
            "balance": "1.0",
            "received": "2.0"
        }
    }
    mocker.patch('requests.get', return_value=mock_response)
    
    balance, received = wallet.check_balance(list(wallet.addresses.values())[0])
    assert balance == 1.0
    assert received == 2.0
