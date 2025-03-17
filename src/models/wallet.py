# Ensure the custom sha3 module is available
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import functools
from hdwallet import HDWallet
from hdwallet.symbols import BTC
from .config import Config
import json
import requests
import logging
from cachetools import TTLCache, cached

class BTCWallet:
    # Change to use a class-level cache instead of a decorator
    _wallet_cache = {}
    _MAX_CACHE_SIZE = 128
    
    @classmethod
    def create_from_mnemonic(cls, mnemonic: str, config: Config):
        """Factory method with caching to avoid recreating wallets"""
        # Use mnemonic as key, not config (which is unhashable)
        cache_key = mnemonic
        if cache_key in cls._wallet_cache:
            return cls._wallet_cache[cache_key]
        
        # Create new instance
        wallet = cls(mnemonic, config)
        
        # Manage cache size
        if len(cls._wallet_cache) >= cls._MAX_CACHE_SIZE:
            # Remove oldest entry (first key)
            cls._wallet_cache.pop(next(iter(cls._wallet_cache)))
            
        # Add to cache
        cls._wallet_cache[cache_key] = wallet
        return wallet
    
    def __init__(self, mnemonic: str, config: Config):
        self.config = config
        # Initialize HDWallet with Bitcoin
        hdwallet = HDWallet(symbol=BTC)
        hdwallet.from_mnemonic(mnemonic)
        hdwallet.from_path(config.derivation_path)
        dumped_data = hdwallet.dumps()  # Store once
        
        # Handle potential missing keys safely and efficiently
        self.addresses = {}
        if 'addresses' in dumped_data:
            for addr_type in config.address_types:
                if addr_type in dumped_data['addresses']:
                    self.addresses[addr_type] = dumped_data['addresses'][addr_type]
        elif 'address' in dumped_data:  # Handle alternative format
            self.addresses = {config.address_types[0]: dumped_data['address']}
            
        # Store only needed data
        self.wif = dumped_data.get('wif', '')
        self.seed = mnemonic  # Store original mnemonic
        self.entropy = dumped_data.get('entropy', '')
        self.private_key = dumped_data.get('private_key', '')

    # Use address as cache key for balance checks, not the whole wallet
    @cached(cache=TTLCache(maxsize=1024, ttl=300))  # Cache for 5 minutes
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
