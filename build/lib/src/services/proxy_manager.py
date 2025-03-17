import logging
from typing import Optional, List
import requests
from urllib.parse import urlparse
import socks
import socket
import random

class ProxyManager:
    def __init__(self, config):
        self.config = config
        self.proxy_scores = {}
        self.dead_proxies = set()
    
    def get_best_proxy(self) -> str:
        if not self.config.proxies:
            return None
        
        valid_proxies = [p for p in self.config.proxies 
                         if self.proxy_scores.get(p, 0) >= self.config.proxy_score_threshold]
        if not valid_proxies:
            return None
        
        return random.choice(valid_proxies)

    def score_proxy(self, proxy: str, success: bool):
        if proxy not in self.proxy_scores:
            self.proxy_scores[proxy] = 0
        if success:
            self.proxy_scores[proxy] = min(self.proxy_scores[proxy] + 1, 10)
        else:
            self.proxy_scores[proxy] = max(self.proxy_scores[proxy] - 2, -10)
