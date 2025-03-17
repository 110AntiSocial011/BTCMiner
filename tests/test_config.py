import pytest
import os
import json
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.config import Config

def test_config_load(setup_test_env):
    config = Config.load()
    assert config.strength == 128
    assert config.language == "english"
    assert config.address_type == "p2pkh"
