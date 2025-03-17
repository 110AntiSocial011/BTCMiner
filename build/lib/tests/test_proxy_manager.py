import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.proxy_manager import ProxyManager

def test_proxy_scoring():
    config = type('Config', (), {'proxies': ['proxy1', 'proxy2'], 'proxy_score_threshold': 5})()
    manager = ProxyManager(config)
    
    manager.score_proxy('proxy1', True)  # Success
    manager.score_proxy('proxy2', False)  # Failure
    
    assert manager.proxy_scores['proxy1'] == 1
    assert manager.proxy_scores['proxy2'] == -2

def test_best_proxy_selection():
    config = type('Config', (), {'proxies': ['proxy1', 'proxy2'], 'proxy_score_threshold': 5})()
    manager = ProxyManager(config)
    
    # Make proxy1 score high
    for _ in range(6):
        manager.score_proxy('proxy1', True)
    
    best_proxy = manager.get_best_proxy()
    assert best_proxy == 'proxy1'
