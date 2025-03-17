# Ensure the custom sha3 module is available
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hdwallet import HDWallet
from hdwallet.symbols import BTC
from .config import Config
import json
import requests
import logging

class BTCWallet:
    def __init__(self, mnemonic: str, config: Config):
        self.config = config
        # Initialize HDWallet with Bitcoin
        hdwallet = HDWallet(symbol=BTC)
        hdwallet.from_mnemonic(mnemonic)
        hdwallet.from_path(config.derivation_path)
        dumped_data = hdwallet.dumps()  # Store once
        self.addresses = {
            addr_type: dumped_data['addresses'][addr_type]
            for addr_type in config.address_types
        }
        self.wif = dumped_data['wif']
        self.seed = mnemonic
        self.entropy = dumped_data['entropy']
        self.private_key = dumped_data['private_key']

    def check_balance(self, address: str) -> tuple[float, float]:
        try:
            response = requests.get(
                f"{self.config.api_url}/{address}",
                timeout=30
            ).json()
            balance = float(response[self.config.api_data_path][self.config.api_balance_path])
            received = float(response[self.config.api_data_path][self.config.api_received_path])
            return balance, received
        except requests.RequestException as e:
            logging.error(f"Error checking balance for {address}: {e}")
            return 0.0, 0.0
