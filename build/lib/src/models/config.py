from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import os
import multiprocessing as mp

@dataclass
class Config:
    strength: int
    language: str 
    passphrase: str = None
    checker_file: str = "check.txt"
    failed_file: str = "failed.txt"
    success_file: str = "success.txt"
    address_type: str = "p2pkh"
    api_url: str = "https://chain.api.btc.com/v3/address"
    api_data_path: str = "data"
    api_balance_path: str = "balance"
    api_received_path: str = "received"
    max_threads: int = 4
    address_types: List[str] = None
    derivation_path: str = "m/44'/0'/0'/0/0"  # Add default derivation path

    def __post_init__(self):
        if self.address_types is None:
            self.address_types = [self.address_type]

    @classmethod
    def load(cls) -> 'Config':
        config_file = os.getenv('CONFIG_FILE', 'config.json')
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found")
            
        with open(config_file) as f:
            data = json.load(f)
            return cls(
                strength=data["settings"]["bruteforcer"]["strength"],
                language=data["settings"]["bruteforcer"]["language"],
                passphrase=data["settings"]["bruteforcer"].get("passphrase"),
                checker_file=data["settings"]["checker"]["filename"],
                failed_file=data["settings"]["general"]["failed"],
                success_file=data["settings"]["general"]["success"],
                address_type=data["settings"]["general"]["addresstype"],
                derivation_path=data["settings"]["general"].get("derivation_path", "m/44'/0'/0'/0/0") # Load derivation path
            )
