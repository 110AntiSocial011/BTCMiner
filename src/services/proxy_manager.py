import logging
from typing import Optional, List, Dict
import requests
from urllib.parse import urlparse
import socks
import socket
import random
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import heapq

class ProxyManager:
    def __init__(self, config):
        self.config = config
        self.proxy_scores = {}
        self.dead_proxies = set()
        self.last_used = {}  # Track last used time
        self.usage_count = {}  # Track usage count
        self.proxy_latencies = {}  # Track response times
        self.proxy_lock = asyncio.Lock()
        
        # Load balancing state
        self.proxy_pool = []
        if self.config.proxies:
            self._initialize_proxy_pool()
    
    def _initialize_proxy_pool(self):
        """Initialize proxy pool with scores and weights"""
        self.proxy_pool = []
        for proxy in self.config.proxies:
            if proxy not in self.dead_proxies:
                # Initial score of 5 (middle range)
                self.proxy_scores[proxy] = 5
                self.usage_count[proxy] = 0
                self.last_used[proxy] = 0
                self.proxy_latencies[proxy] = 1.0  # Default 1 second
    
    async def get_best_proxy(self) -> str:
        """Get the best proxy based on multiple factors"""
        if not self.config.proxies:
            return None
        
        # Use aync lock to prevent race conditions
        async with self.proxy_lock:
            current_time = time.time()
            
            # Calculate a composite score for each proxy
            proxy_rankings = []
            for proxy in self.config.proxies:
                if proxy in self.dead_proxies:
                    continue
                    
                # Skip if proxy is below threshold
                if self.proxy_scores.get(proxy, 0) < self.config.proxy_score_threshold:
                    continue
                
                # Time since last use (prefer proxies that haven't been used recently)
                time_factor = min(current_time - self.last_used.get(proxy, 0), 30) / 30
                
                # Usage count (prefer less used proxies)
                usage_factor = 1.0 / (1.0 + self.usage_count.get(proxy, 0) * 0.01)
                
                # Latency factor (prefer faster proxies)
                latency_factor = 1.0 / max(self.proxy_latencies.get(proxy, 1.0), 0.1)
                
                # Score factor (prefer higher scored proxies)
                score_factor = self.proxy_scores.get(proxy, 0) / 10
                
                # Composite score (weighted sum)
                composite_score = (
                    0.3 * time_factor + 
                    0.2 * usage_factor + 
                    0.3 * latency_factor + 
                    0.2 * score_factor
                )
                
                # Add to ranking
                heapq.heappush(proxy_rankings, (-composite_score, proxy))
            
            if not proxy_rankings:
                return None
            
            # Get the best proxy
            _, best_proxy = heapq.heappop(proxy_rankings)
            
            # Update tracking
            self.last_used[best_proxy] = current_time
            self.usage_count[best_proxy] = self.usage_count.get(best_proxy, 0) + 1
            
            return best_proxy

    async def track_proxy_performance(self, proxy: str, success: bool, latency: float = None):
        """Update proxy performance metrics"""
        if not proxy:
            return
            
        async with self.proxy_lock:
            # Update score
            if proxy not in self.proxy_scores:
                self.proxy_scores[proxy] = 5  # Middle ground initial score
                
            if success:
                self.proxy_scores[proxy] = min(self.proxy_scores[proxy] + 1, 10)
            else:
                self.proxy_scores[proxy] = max(self.proxy_scores[proxy] - 2, -10)
                
            # Update latency if provided
            if latency is not None:
                if proxy not in self.proxy_latencies:
                    self.proxy_latencies[proxy] = latency
                else:
                    # Exponential moving average (more weight to recent measurements)
                    self.proxy_latencies[proxy] = (
                        0.8 * self.proxy_latencies[proxy] + 
                        0.2 * latency
                    )
            
            # Add to dead proxies if score is too low
            if self.proxy_scores[proxy] <= -5:
                self.dead_proxies.add(proxy)
    
    # Add a non-async version for compatibility with tests
    def score_proxy(self, proxy: str, success: bool):
        """Synchronous version of track_proxy_performance for backwards compatibility"""
        if not proxy:
            return
            
        # Update score
        if proxy not in self.proxy_scores:
            self.proxy_scores[proxy] = 5
            
        if success:
            self.proxy_scores[proxy] = min(self.proxy_scores[proxy] + 1, 10)
        else:
            self.proxy_scores[proxy] = max(self.proxy_scores[proxy] - 2, -10)
        
        # Add to dead proxies if score is too low
        if self.proxy_scores[proxy] <= -5:
            self.dead_proxies.add(proxy)
    
    async def check_proxy_health(self, proxy: str) -> bool:
        """Check if a proxy is working"""
        try:
            timeout = aiohttp.ClientTimeout(total=self.config.proxy_timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                start_time = time.time()
                async with session.get(
                    "http://www.google.com", 
                    proxy=proxy,  # Changed from proxies=proxy
                    allow_redirects=False
                ) as response:
                    latency = time.time() - start_time
                    success = 200 <= response.status < 300
                    
                    # Track performance
                    await self.track_proxy_performance(proxy, success, latency)
                    return success
                    
        except Exception:
            await self.track_proxy_performance(proxy, False)
            return False
    
    async def refresh_proxies(self):
        """Check all proxies and refresh the list"""
        if not self.config.proxies:
            return
            
        # Create tasks to check all proxies in parallel
        tasks = [self.check_proxy_health(proxy) for proxy in self.config.proxies]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Reset dead proxies after checking all
        self.dead_proxies = {
            proxy for proxy, result in zip(self.config.proxies, results)
            if not isinstance(result, bool) or not result
        }
        
        # Reinitialize pool with active proxies
        self._initialize_proxy_pool()
        
        return len(self.config.proxies) - len(self.dead_proxies)
