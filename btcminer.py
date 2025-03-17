import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Set, List, Dict, Callable
import csv
import sqlite3
import pandas as pd
from itertools import cycle
import aiohttp
import asyncio
from memory_profiler import profile
from concurrent.futures import ThreadPoolExecutor
import time
from ratelimit import limits, sleep_and_retry
import threading
from tqdm import tqdm
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import backoff
import gc
import multiprocessing as mp
from functools import lru_cache
import websockets
import ujson
import redis.asyncio as aioredis 
from redis.asyncio import ConnectionPool, Sentinel
from prometheus_client import Histogram
import requests
from colorama import Fore, init
from hdwallet import HDWallet
from mnemonic import Mnemonic
import aioh2
from prometheus_client import start_http_server, Counter, Gauge
import aiosqlite
from contextlib import asynccontextmanager
import psutil
import importlib
import queue
import pika
import socks
import socket
from urllib.parse import urlparse
import hashlib
import subprocess
import shutil
import tempfile
from tkinter import Tk, Label, Button, Text, Scrollbar, END
import random
import functools
import concurrent.futures
from cachetools import TTLCache, cached

# Initialize colorama
init()

# Constants
CONFIG_FILE = "config.json"
DEFAULT_API_TIMEOUT = 30

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add Prometheus metrics
ADDRESSES_CHECKED = Counter('addresses_checked_total', 'Total addresses checked')
BALANCE_HITS = Counter('balance_hits_total', 'Total addresses with balance')
API_ERRORS = Counter('api_errors_total', 'Total API errors')
ACTIVE_THREADS = Gauge('active_threads', 'Number of active checking threads')
BALANCE_CHECK_DURATION = Histogram('balance_check_duration_seconds', 'Time spent checking balance')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
API_REQUESTS = Counter('api_requests_total', 'Total API requests')
API_LATENCY = Histogram('api_latency_seconds', 'API request latency')
VALID_ADDRESSES = Counter('valid_addresses_total', 'Total valid addresses')
INVALID_ADDRESSES = Counter('invalid_addresses_total', 'Total invalid addresses')

@dataclass
class Config:
    strength: int
    language: str 
    passphrase: str
    checker_file: str
    failed_file: str
    success_file: str
    address_type: str
    api_url: str
    api_data_path: str
    api_balance_path: str
    api_received_path: str
    max_threads: int = 4
    requests_per_minute: int = 60
    max_retries: int = 3
    derivation_path: str = "m/44'/0'/0'/0/0"
    proxies: List[str] = None
    batch_size: int = 1000
    checkpoint_interval: int = 5000
    export_format: str = "csv"
    database_file: str = "wallets.db"
    email_notifications: bool = False
    email_recipients: List[str] = None
    smtp_server: str = ""
    smtp_port: int = 0
    smtp_username: str = ""
    smtp_password: str = ""
    cryptocurrencies: List[str] = ["BTC"]
    logging_level: str = "INFO"
    max_addresses: int = 10000
    api_headers: Dict[str, str] = None
    ban_detection_string: str = None
    auto_proxy_switch: bool = False
    address_types: List[str] = None
    log_interval: int = 600  # Log statistics every 10 minutes
    redis_url: str = "redis://localhost"
    redis_ttl: int = 3600
    http2_enabled: bool = True
    metrics_port: int = 9090
    db_pool_size: int = 5
    proxy_providers: List[str] = None
    paths_per_coin: Dict[str, str] = None
    websocket_port: int = 8765
    enable_realtime_updates: bool = False
    redis_pool_size: int = 10
    redis_sentinel_nodes: List[str] = None
    custom_address_generators: Dict[str, str] = None
    process_count: int = None
    gc_interval: int = 300
    memory_limit: int = None  # in MB
    api_keys: List[str] = None
    api_key_rotation_interval: int = 3600
    address_generator_plugins: List[str] = None
    tor_proxy: str = None
    proxy_check_interval: int = 3600
    proxy_timeout: int = 10
    message_queue_url: str = None
    api_usage_log_interval: int = 600
    verify_address: bool = True
    proxy_score_threshold: int = 5
    api_endpoints: Dict[str, str] = None
    auto_update: bool = False
    update_url: str = "https://github.com/LizardX2/BTCMiner"
    checkpoint_file: str = "checkpoint.json"
    dynamic_batch_size: bool = True
    min_batch_size: int = 100
    max_batch_size: int = 10000
    memory_log_interval: int = 300

    @classmethod
    def load(cls) -> 'Config':
        if not os.path.isfile(CONFIG_FILE):
            raise FileNotFoundError(f"Config file {CONFIG_FILE} not found")
            
        with open(CONFIG_FILE) as f:
            data = json.load(f)
            return cls(
                strength=data["settings"]["bruteforcer"]["strength"], # Fixed typo
                language=data["settings"]["bruteforcer"]["language"],
                passphrase=data["settings"]["bruteforcer"]["passphrase"], # Fixed typo
                checker_file=data["settings"]["checker"]["filename"],
                failed_file=data["settings"]["general"]["failed"],
                success_file=data["settings"]["general"]["success"],
                address_type=data["settings"]["general"]["addresstype"],
                api_url=data["settings"]["general"]["api"]["api_url"],
                api_data_path=data["settings"]["general"]["api"]["api_get_data"],
                api_balance_path=data["settings"]["general"]["api"]["api_get_balance"],
                api_received_path=data["settings"]["general"]["api"]["api_get_received"], # Fixed typo
                max_threads=data["settings"]["general"].get("max_threads", 4),
                requests_per_minute=data["settings"]["general"].get("requests_per_minute", 60),
                max_retries=data["settings"]["general"].get("max_retries", 3),
                derivation_path=data["settings"]["general"].get("derivation_path", "m/44'/0'/0'/0/0"),
                proxies=data["settings"]["general"].get("proxies", []),
                batch_size=data["settings"]["general"].get("batch_size", 1000),
                checkpoint_interval=data["settings"]["general"].get("checkpoint_interval", 5000),
                export_format=data["settings"]["general"].get("export_format", "csv"),
                database_file=data["settings"]["general"].get("database_file", "wallets.db"),
                email_notifications=data["settings"]["general"].get("email_notifications", False),
                email_recipients=data["settings"]["general"].get("email_recipients", []),
                smtp_server=data["settings"]["general"].get("smtp_server", ""),
                smtp_port=data["settings"]["general"].get("smtp_port", 0),
                smtp_username=data["settings"]["general"].get("smtp_username", ""),
                smtp_password=data["settings"]["general"].get("smtp_password", ""),
                cryptocurrencies=data["settings"]["general"].get("cryptocurrencies", ["BTC"]),
                logging_level=data["settings"]["general"].get("logging_level", "INFO"),
                max_addresses=data["settings"]["general"].get("max_addresses", 10000),
                api_headers=data["settings"]["general"].get("api_headers", {}),
                ban_detection_string=data["settings"]["general"].get("ban_detection_string", None),
                auto_proxy_switch=data["settings"]["general"].get("auto_proxy_switch", False),
                address_types=data["settings"]["general"].get("address_types", ["p2pkh"]),
                log_interval=data["settings"]["general"].get("log_interval", 600),
                redis_url=data["settings"]["general"].get("redis_url", "redis://localhost"),
                redis_ttl=data["settings"]["general"].get("redis_ttl", 3600),
                http2_enabled=data["settings"]["general"].get("http2_enabled", True),
                metrics_port=data["settings"]["general"].get("metrics_port", 9090),
                db_pool_size=data["settings"]["general"].get("db_pool_size", 5),
                proxy_providers=data["settings"]["general"].get("proxy_providers", []),
                paths_per_coin=data["settings"]["general"].get("paths_per_coin", {}),
                websocket_port=data["settings"]["general"].get("websocket_port", 8765),
                enable_realtime_updates=data["settings"]["general"].get("enable_realtime_updates", False),
                redis_pool_size=data["settings"]["general"].get("redis_pool_size", 10),
                redis_sentinel_nodes=data["settings"]["general"].get("redis_sentinel_nodes", []),
                custom_address_generators=data["settings"]["general"].get("custom_address_generators", {}),
                process_count=data["settings"]["general"].get("process_count", mp.cpu_count()),
                gc_interval=data["settings"]["general"].get("gc_interval", 300),
                memory_limit=data["settings"]["general"].get("memory_limit", None),
                api_keys=data["settings"]["general"].get("api_keys", []),
                api_key_rotation_interval=data["settings"]["general"].get("api_key_rotation_interval", 3600),
                address_generator_plugins=data["settings"]["general"].get("address_generator_plugins", []),
                tor_proxy=data["settings"]["general"].get("tor_proxy", None),
                proxy_check_interval=data["settings"]["general"].get("proxy_check_interval", 3600),
                proxy_timeout=data["settings"]["general"].get("proxy_timeout", 10),
                message_queue_url=data["settings"]["general"].get("message_queue_url", None),
                api_usage_log_interval=data["settings"]["general"].get("api_usage_log_interval", 600),
                verify_address=data["settings"]["general"].get("verify_address", True),
                proxy_score_threshold=data["settings"]["general"].get("proxy_score_threshold", 5),
                api_endpoints=data["settings"]["general"].get("api_endpoints", {}),
                auto_update=data["settings"]["general"].get("auto_update", False),
                update_url=data["settings"]["general"].get("update_url", "https://github.com/LizardX2/BTCMiner"),
                checkpoint_file=data["settings"]["general"].get("checkpoint_file", "checkpoint.json"),
                dynamic_batch_size=data["settings"]["general"].get("dynamic_batch_size", True),
                min_batch_size=data["settings"]["general"].get("min_batch_size", 100),
                max_batch_size=data["settings"]["general"].get("max_batch_size", 10000),
                memory_log_interval=data["settings"]["general"].get("memory_log_interval", 300)
            )

    def validate(self):
        strength_options = {128, 160, 192, 224, 256}
        language_options = {"english", "french", "italian", "spanish", "chinese_simplified", "chinese_traditional", "japanese", "korean"}
        address_options = {"p2pkh", "p2sh", "p2wpkh", "p2wpkh_in_p2sh", "p2wsh", "p2wsh_in_p2sh"}

        if self.strength not in strength_options:
            raise ValueError(f"Invalid strength option: {self.strength}")
        if self.language not in language_options:
            raise ValueError(f"Invalid language option: {self.language}")
        if self.address_type not in address_options:
            raise ValueError(f"Invalid address type: {self.address_type}")
        if not all(addr_type in address_options for addr_type in self.address_types):
            raise ValueError(f"Invalid address types: {self.address_types}")
        if self.email_notifications and (not self.email_recipients or not self.smtp_server or not self.smtp_port or not self.smtp_username or not self.smtp_password):
            raise ValueError("Email notifications are enabled, but email settings are not properly configured.")

class AddressGenerator:
    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self.module = self._load_custom_generator(algorithm)

    def _load_custom_generator(self, algorithm: str) -> Callable:
        try:
            module = __import__(f"generators.{algorithm}", fromlist=['generate'])
            return module.generate
        except ImportError:
            logging.warning(f"Custom generator {algorithm} not found, using default")
            return None

    @lru_cache(maxsize=1000)
    def generate_address(self, seed: str) -> str:
        if self.module:
            return self.module(seed)
        return None

class BTCWallet:
    @classmethod
    @functools.lru_cache(maxsize=128)
    def create_from_mnemonic(cls, mnemonic: str, config: Config):
        """Factory method with caching to avoid recreating wallets"""
        return cls(mnemonic, config)
        
    def __init__(self, mnemonic: str, config: Config):
        self.config = config
        hdwallet = HDWallet(symbol="BTC", use_default_path=False)
        hdwallet.from_mnemonic(mnemonic=mnemonic)
        hdwallet.from_path(config.derivation_path)
        dumped_data = hdwallet.dumps()
        self.addresses = {addr_type: dumped_data['addresses'][addr_type] for addr_type in config.address_types}
        self.wif = dumped_data.get('wif', '')
        self.seed = mnemonic
        self.entropy = dumped_data.get('entropy', '')
        self.private_key = dumped_data.get('private_key', '')

    @cached(cache=TTLCache(maxsize=1024, ttl=300))  # Cache balance checks for 5 minutes
    def check_balance(self, address: str) -> tuple[float, float]:
        try:
            response = requests.get(
                f"{self.config.api_url}/{address}",
                timeout=DEFAULT_API_TIMEOUT
            ).json()
            balance = float(response[self.config.api_data_path][self.config.api_balance_path])
            received = float(response[self.config.api_data_path][self.config.api_received_path])
            return balance, received
        except requests.RequestException as e:
            logging.error(f"Error checking balance for {address}: {e}")
            return 0.0, 0.0

    def save_result(self, address: str, balance: float, received: float):
        result = {
            "address": address,
            "balance": balance,
            "received": received,
            "seed": self.seed,
            "private_key": self.private_key,
            "entropy": self.entropy,
            "wif": self.wif
        }
        filename = self.config.success_file if (balance > 0 or received > 0) else self.config.failed_file
        with open(filename, "a") as f:
            f.write(json.dumps(result) + "\n")

    def __str__(self) -> str:
        return f"{self.addresses} | {self.seed} | {self.private_key}"

class BTCMiner:
    def __init__(self):
        self.config = Config.load()
        self.config.validate()
        logging.getLogger().setLevel(self.config.logging_level)
        self.session = requests.Session()
        self._validate_files()
        self.checked_addresses: Set[str] = set()
        self.stats = {"checked": 0, "hits": 0, "errors": 0}
        self.stats_lock = threading.Lock()
        self.proxy_pool = cycle(self.config.proxies) if self.config.proxies else None
        self.db_conn = self._init_database()
        self.checkpoint = self._load_checkpoint()
        self.pause_event = threading.Event()
        self.current_proxy = None
        if self.config.proxies:
            self.proxy_pool = cycle(self.config.proxies)
            self.current_proxy = next(self.proxy_pool)
        else:
            self.proxy_pool = None
        self.log_timer = threading.Timer(self.config.log_interval, self.log_statistics)
        self.log_timer.start()
        self.redis = None
        self.db_pool = None
        self.websocket_server = None
        self.connected_clients = set()
        self.redis_pool = None
        self.address_generators = {}
        self.process_pool = None
        self.last_gc = time.time()
        self.api_key_index = 0
        self.api_key_rotation_timer = threading.Timer(self.config.api_key_rotation_interval, self._rotate_api_key)
        self.api_key_rotation_timer.start()
        self.dead_proxies = set()
        self.proxy_check_timer = threading.Timer(self.config.proxy_check_interval, self._check_proxies)
        self.proxy_check_timer.start()
        self.message_queue_channel = None
        self.api_usage = {"requests": 0, "errors": 0}
        self.api_usage_lock = threading.Lock()
        self.api_usage_timer = threading.Timer(self.config.api_usage_log_interval, self._log_api_usage)
        self.api_usage_timer.start()
        self.proxy_scores = {}
        if self.config.auto_update:
            self._update_tool()
        
        if self.config.enable_realtime_updates:
            self._start_websocket_server()
        
        if self.config.custom_address_generators:
            self._initialize_address_generators()
        
        if self.config.process_count:
            self.process_pool = mp.Pool(self.config.process_count)
        start_http_server(self.config.metrics_port)
        self.batch_size = self.config.batch_size
        self.memory_log_timer = threading.Timer(self.config.memory_log_interval, self._log_memory_usage)
        self.memory_log_timer.start()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, os.cpu_count() * 4),  # More efficient sizing
            thread_name_prefix="btc_miner_io"
        )
        if self.config.process_count:
            self.process_pool = mp.Pool(
                processes=min(self.config.process_count, os.cpu_count()),
                maxtasksperchild=100  # Prevent memory leaks
            )
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=100, 
            pool_maxsize=100,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    async def initialize(self):
        self.redis = await aioredis.from_url(self.config.redis_url)
        self.db_pool = await aiosqlite.create_pool(
            self.config.database_file,
            min_size=1,
            max_size=self.config.db_pool_size
        )
        if self.config.redis_sentinel_nodes:
            sentinel = Sentinel(self.config.redis_sentinel_nodes)
            self.redis = await sentinel.master_for(
                'mymaster',
                pool_size=self.config.redis_pool_size
            )
        else:
            self.redis_pool = ConnectionPool.from_url(
                self.config.redis_url,
                max_connections=self.config.redis_pool_size
            )
            self.redis = await aioredis.Redis.from_pool(self.redis_pool)
        if self.config.address_generator_plugins:
            self._load_address_generator_plugins()
        if self.config.message_queue_url:
            await self._connect_to_message_queue()

    @asynccontextmanager
    async def get_db(self):
        async with self.db_pool.acquire() as conn:
            yield conn

    def _init_database(self):
        conn = sqlite3.connect(self.config.database_file, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")  # Use Write-Ahead Logging
        conn.execute("PRAGMA synchronous=NORMAL")  # Less fsync, better performance
        conn.execute("PRAGMA cache_size=10000")  # Larger cache
        conn.execute("PRAGMA temp_store=MEMORY")  # Store temp files in memory
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wallets (
                address TEXT PRIMARY KEY,
                balance REAL,
                received REAL,
                seed TEXT,
                private_key TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_checked DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_balance ON wallets(balance)")
        return conn

    def _load_checkpoint(self) -> int:
        try:
            with open(self.config.checkpoint_file, 'r') as f:
                return json.load(f)['last_checked']
        except FileNotFoundError:
            return 0

    def _save_checkpoint(self, count: int):
        with open(self.config.checkpoint_file, 'w') as f:
            json.dump({'last_checked': count}, f)

    def _switch_proxy(self):
        if self.proxy_pool:
            self.current_proxy = next(self.proxy_pool)
            logging.info(f"Switched proxy to {self.current_proxy}")
        else:
            logging.warning("No proxies available to switch to.")

    def pause(self):
        self.pause_event.set()
        logging.info("Bruteforce paused. Press 'r' to resume.")

    def resume(self):
        self.pause_event.clear()
        logging.info("Bruteforce resumed.")

    def _check_pause(self):
        if self.pause_event.is_set():
            logging.info("Bruteforce is paused. Waiting to resume...")
            self.pause_event.wait()
            logging.info("Bruteforce resumed.")

    async def _check_address_cached(self, address: str) -> Optional[tuple[float, float]]:
        if self.redis:
            cached = await self.redis.get(address)
            if cached:
                return tuple(map(float, cached.split(':')))
        return None

    async def _cache_address(self, address: str, balance: float, received: float):
        if self.redis:
            await self.redis.set(
                address,
                f"{balance}:{received}",
                expire=self.config.redis_ttl
            )

    @profile
    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, aioh2.ConnectionError), max_tries=5)
    @BALANCE_CHECK_DURATION.time()
    async def _check_address_async(self, wallet: BTCWallet, address: str) -> None:
        if self.config.verify_address:
            if not self._is_valid_address(address):
                INVALID_ADDRESSES.inc()
                logging.warning(f"Invalid address: {address}")
                return
            VALID_ADDRESSES.inc()

        self._check_memory_usage()
        self._adjust_threads()
        self._adjust_batch_size()
        if address in self.checked_addresses:
            return

        cached_result = await self._check_address_cached(address)
        if cached_result:
            balance, received = cached_result
            await self._process_result(wallet, address, balance, received)
            return

        proxy = self._get_best_proxy()
        headers = self.config.api_headers
        headers["Authorization"] = f"Bearer {self._get_current_api_key()}"

        start_time = time.time()
        success = False
        try:
            if self.config.tor_proxy:
                session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False), headers=headers)
                session._connector._ssl = False
                async with session.get(f"{self.config.api_url}/{address}", proxy=self.config.tor_proxy, timeout=DEFAULT_API_TIMEOUT) as response:
                    response.raise_for_status()
                    data = await response.text()
            elif self.config.http2_enabled:
                client = aioh2.HttpClient(proxy=proxy)
                try:
                    response = await client.request(
                        'GET',
                        f"{self.config.api_url}/{address}",
                        headers=headers
                    )
                    data = await response.read()
                finally:
                    await client.close()
            else:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(
                        f"{self.config.api_url}/{address}",
                        proxy=proxy,
                        timeout=DEFAULT_API_TIMEOUT
                    ) as response:
                        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                        data = await response.text()

            try:
                json_data = json.loads(data)
                balance = float(json_data[self.config.api_data_path][self.config.api_balance_path])
                received = float(json_data[self.config.api_data_path][self.config.api_received_path])
                success = True
            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"Error parsing API response: {e}. Response: {data}")
                return
        except Exception as e:
            logging.error(f"API request failed: {e}")
            return
        finally:
            latency = time.time() - start_time
            API_LATENCY.observe(latency)
            with self.api_usage_lock:
                self.api_usage["requests"] += 1
            self._score_proxy(proxy, success)

        # Process response and update metrics
        ADDRESSES_CHECKED.inc()
        if balance > 0 or received > 0:
            BALANCE_HITS.inc()
        
        await self._cache_address(address, balance, received)
        await self._process_result(wallet, address, balance, received)

    async def _process_result(self, wallet: BTCWallet, address: str, balance: float, received: float):
        async with self.get_db() as db:
            await db.execute(
                "INSERT OR REPLACE INTO wallets (address, balance, received, seed, private_key) VALUES (?, ?, ?, ?, ?)",
                (address, balance, received, wallet.seed, wallet.private_key)
            )
            await db.commit()

        wallet.save_result(address, balance, received)
        self._update_stats(balance, received)
        if balance > 0 or received > 0:
            await self._send_email_notification_async(wallet, address, balance, received)
            await self._broadcast_update({
                'address': address,
                'balance': balance,
                'received': received
            })
            self._publish_to_message_queue(ujson.dumps({
                'address': address,
                'balance': balance,
                'received': received,
                'seed': wallet.seed
            }))

    async def cleanup(self):
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
        if self.db_pool:
            await self.db_pool.close()
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        if self.process_pool:
            self.process_pool.close()
            self.process_pool.join()
        if self.redis_pool:
            self.redis_pool.disconnect()
        if self.message_queue_channel:
            self.message_queue_channel.close()
        self.proxy_check_timer.cancel()
        self.api_usage_timer.cancel()
        self.memory_log_timer.cancel()

    async def update_proxies(self):
        """Fetch fresh proxies from providers"""
        if not self.config.proxy_providers:
            return
        
        async with aiohttp.ClientSession() as session:
            for provider in self.config.proxy_providers:
                try:
                    async with session.get(provider) as response:
                        proxies = await response.text()
                        self.config.proxies.extend(proxies.splitlines())
                except Exception as e:
                    logging.error(f"Failed to fetch proxies from {provider}: {e}")

        if self.config.proxies:
            self.proxy_pool = cycle(self.config.proxies)
            self.current_proxy = next(self.proxy_pool)

    async def bruteforce(self):
        batch = []
        count = self._load_checkpoint()
        
        semaphore = asyncio.Semaphore(50)  # Limit concurrent API requests
        
        with tqdm(desc="Bruteforcing", unit="addr") as pbar:
            while count < self.config.max_addresses:
                batch = await self._generate_batch(self.batch_size)
                
                async with asyncio.TaskGroup() as tg:
                    for wallet, address in batch:
                        tg.create_task(self._check_address_with_semaphore(semaphore, wallet, address))
                
                count += len(batch)
                pbar.update(len(batch))
                pbar.set_postfix(self.stats)
                
                if count % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(count)
                    self._export_results()
                
                batch.clear()
                if count % 10000 == 0:
                    gc.collect()

    async def _check_address_with_semaphore(self, semaphore: asyncio.Semaphore, wallet: BTCWallet, address: str):
        """Process an address check with controlled concurrency"""
        async with semaphore:
            await self._check_address_async(wallet, address)

    async def _generate_batch(self, size: int) -> list:
        """Generate a batch of wallets and addresses more efficiently"""
        batch = []
        mnemonics = await asyncio.get_event_loop().run_in_executor(
            self.process_pool,
            lambda: [
                Mnemonic(self.config.language).generate(strength=self.config.strength)
                for _ in range(size)
            ]
        )
        
        for mnemonic in mnemonics:
            wallet = BTCWallet.create_from_mnemonic(mnemonic, self.config)
            batch.extend((wallet, addr) for addr in wallet.addresses.values())
        
        return batch

    def _export_results(self):
        query = "SELECT * FROM wallets"
        df = pd.read_sql_query(query, self.db_conn)
        
        if self.config.export_format == "csv":
            df.to_csv("results.csv", index=False)
        elif self.config.export_format == "excel":
            df.to_excel("results.xlsx", index=False)
        elif self.config.export_format == "json":
            df.to_json("results.json", orient="records")

    def log_statistics(self):
        logging.info(f"Statistics: {self.stats}")
        self.log_timer = threading.Timer(self.config.log_interval, self.log_statistics)
        self.log_timer.start()

    def _send_email_notification(self, wallet: BTCWallet, address: str, balance: float, received: float):
        if not self.config.email_notifications:
            return
        
        subject = "BTCMiner Hit Notification"
        body = f"Address: {address}\nBalance: {balance}\nReceived: {received}\nSeed: {wallet.seed}\nPrivate Key: {wallet.private_key}"
        
        msg = MIMEMultipart()
        msg['From'] = self.config.smtp_username
        msg['To'] = ", ".join(self.config.email_recipients)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.smtp_username, self.config.smtp_password)
            text = msg.as_string()
            server.sendmail(self.config.smtp_username, self.config.email_recipients, text)
            server.quit()
            logging.info("Email notification sent successfully")
        except Exception as e:
            logging.error(f"Failed to send email notification: {e}")

    @sleep_and_retry
    @limits(calls=60, period=60)
    def _check_address(self, wallet: BTCWallet, address: str) -> None:
        if address in self.checked_addresses:
            return
            
        for attempt in range(self.config.max_retries):
            try:
                balance, received = wallet.check_balance(address)
                wallet.save_result(address, balance, received)
                self._log_result(wallet, address, balance)
                
                with self.stats_lock:
                    self.stats["checked"] += 1
                    if balance > 0 or received > 0:
                        self.stats["hits"] += 1
                        self._send_email_notification(wallet, address, balance, received)
                        
                self.checked_addresses.add(address)
                return
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    with self.stats_lock:
                        self.stats["errors"] += 1
                    logging.error(f"Failed to check {address} after {self.config.max_retries} attempts: {e}")
                time.sleep(1)

    def check_from_file(self):
        with open(self.config.checker_file) as f:
            mnemonics = f.readlines()
            
        with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            with tqdm(total=len(mnemonics), desc="Checking", unit="addr") as pbar:
                futures = []
                for mnemonic in mnemonics:
                    wallet = BTCWallet(mnemonic.strip(), self.config)
                    for address in wallet.addresses.values():
                        futures.append(executor.submit(self._check_address, wallet, address))
                    
                for future in futures:
                    future.result()
                    pbar.update(1)
                    pbar.set_postfix(self.stats)

    def _validate_files(self):
        required = [self.config.failed_file, self.config.success_file, self.config.checker_file]
        missing = [f for f in required if not os.path.isfile(f)]
        if missing:
            raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

    def _log_result(self, wallet: BTCWallet, address: str, balance: float):
        current = datetime.now().strftime("%H:%M:%S")
        logging.info(f"{address} | BAL: {balance}$ | SEED: {wallet.seed} | PRIV: {wallet.private_key}")

    def test_config(self):
        logging.info("Testing configuration...")
        logging.info(f"Strength: {self.config.strength}")
        logging.info(f"Language: {self.config.language}")
        logging.info(f"Passphrase: {self.config.passphrase}")
        logging.info(f"Checker file: {self.config.checker_file}")
        logging.info(f"Failed file: {self.config.failed_file}")
        logging.info(f"Success file: {self.config.success_file}")
        logging.info(f"Address type: {self.config.address_type}")
        logging.info(f"API URL: {self.config.api_url}")
        logging.info(f"API data path: {self.config.api_data_path}")
        logging.info(f"API balance path: {self.config.api_balance_path}")
        logging.info(f"API received path: {self.config.api_received_path}")
        logging.info(f"Email notifications: {self.config.email_notifications}")
        logging.info(f"Email recipients: {self.config.email_recipients}")
        logging.info(f"SMTP server: {self.config.smtp_server}")
        logging.info(f"SMTP port: {self.config.smtp_port}")
        logging.info(f"SMTP username: {self.config.smtp_username}")
        logging.info(f"Cryptocurrencies: {self.config.cryptocurrencies}")
        logging.info(f"Logging level: {self.config.logging_level}")
        logging.info(f"Max addresses: {self.config.max_addresses}")
        logging.info(f"API Headers: {self.config.api_headers}")
        logging.info(f"Ban Detection String: {self.config.ban_detection_string}")
        logging.info(f"Auto Proxy Switch: {self.config.auto_proxy_switch}")
        logging.info(f"Address types: {self.config.address_types}")
        logging.info(f"Log interval: {self.config.log_interval} seconds")
        logging.info(f"API Keys: {self.config.api_keys}")
        logging.info(f"API Key Rotation Interval: {self.config.api_key_rotation_interval} seconds")
        logging.info(f"Address Generator Plugins: {self.config.address_generator_plugins}")
        logging.info(f"Tor Proxy: {self.config.tor_proxy}")
        logging.info(f"Proxy Check Interval: {self.config.proxy_check_interval}")
        logging.info(f"Proxy Timeout: {self.config.proxy_timeout}")
        logging.info(f"Message Queue URL: {self.config.message_queue_url}")
        logging.info(f"API Usage Log Interval: {self.config.api_usage_log_interval}")
        logging.info(f"Verify Address: {self.config.verify_address}")
        logging.info(f"Proxy Score Threshold: {self.config.proxy_score_threshold}")
        logging.info(f"API Endpoints: {self.config.api_endpoints}")
        logging.info(f"Auto Update: {self.config.auto_update}")
        logging.info(f"Update URL: {self.config.update_url}")
        logging.info(f"Checkpoint File: {self.config.checkpoint_file}")
        logging.info(f"Dynamic Batch Size: {self.config.dynamic_batch_size}")
        logging.info(f"Min Batch Size: {self.config.min_batch_size}")
        logging.info(f"Max Batch Size: {self.config.max_batch_size}")
        logging.info(f"Memory Log Interval: {self.config.memory_log_interval}")
        logging.info("Configuration test completed successfully.")

    async def main(self):
        await self.initialize()
        try:
            mode = input(f"{Fore.YELLOW}[?]{Fore.RESET} Choose mode [C]hecker, [B]ruteforcer, [T]est, or [E]xport: ").lower()
            
            if mode == 'b':
                await self.bruteforce()
            elif mode == 'c':
                self.check_from_file()
            elif mode == 't':
                self.test_config()
            elif mode == 'e':
                self._export_results()
                print(f"{Fore.GREEN}Results exported successfully{Fore.RESET}")
            else:
                print(f"{Fore.RED}Invalid choice. Must be 'C', 'B', 'T' or 'E'{Fore.RESET}")
        finally:
            await self.cleanup()

    async def _start_websocket_server(self):
        async def handler(websocket, path):
            self.connected_clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.connected_clients.remove(websocket)

        self.websocket_server = await websockets.serve(
            handler, 
            "localhost", 
            self.config.websocket_port
        )

    async def _broadcast_update(self, data: dict):
        if not self.connected_clients:
            return
        message = ujson.dumps(data)
        await asyncio.gather(*[
            client.send(message)
            for client in self.connected_clients
        ])

    def _check_memory_usage(self):
        if not self.config.memory_limit:
            return
        
        current_time = time.time()
        if current_time - self.last_gc >= self.config.gc_interval:
            gc.collect()
            self.last_gc = current_time
        
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        MEMORY_USAGE.set(memory_usage)
        
        if memory_usage > self.config.memory_limit:
            logging.warning(f"Memory usage ({memory_usage}MB) exceeded limit ({self.config.memory_limit}MB)")
            gc.collect()

    def _initialize_address_generators(self):
        for name, algorithm in self.config.custom_address_generators.items():
            self.address_generators[name] = AddressGenerator(algorithm)

    def _rotate_api_key(self):
        if self.config.api_keys:
            self.api_key_index = (self.api_key_index + 1) % len(self.config.api_keys)
            logging.info(f"Rotated API key to index {self.api_key_index}")
        self.api_key_rotation_timer = threading.Timer(self.config.api_key_rotation_interval, self._rotate_api_key)
        self.api_key_rotation_timer.start()

    def _get_current_api_key(self) -> str:
        if self.config.api_keys:
            return self.config.api_keys[self.api_key_index]
        return ""

    def _adjust_threads(self):
        load = psutil.cpu_percent(interval=1)
        if load > 80 and self.config.max_threads > 1:
            self.config.max_threads -= 1
            logging.info(f"High CPU load detected ({load}%). Reducing threads to {self.config.max_threads}")
        elif load < 50 and self.config.max_threads < mp.cpu_count():
            self.config.max_threads += 1
            logging.info(f"Low CPU load detected ({load}%). Increasing threads to {self.config.max_threads}")

    def _adjust_batch_size(self):
        load = psutil.cpu_percent(interval=1)
        if load > 80 and self.batch_size > self.config.min_batch_size:
            self.batch_size = max(self.batch_size // 2, self.config.min_batch_size)
            logging.info(f"High CPU load detected ({load}%). Reducing batch size to {self.batch_size}")
        elif load < 50 and self.batch_size < self.config.max_batch_size:
            self.batch_size = min(self.batch_size * 2, self.config.max_batch_size)
            logging.info(f"Low CPU load detected ({load}%). Increasing batch size to {self.batch_size}")

    def _log_memory_usage(self):
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        MEMORY_USAGE.set(memory_usage)
        logging.info(f"Memory usage: {memory_usage} MB")
        self.memory_log_timer = threading.Timer(self.config.memory_log_interval, self._log_memory_usage)
        self.memory_log_timer.start()

    def _load_address_generator_plugins(self):
        for plugin_name in self.config.address_generator_plugins:
            try:
                module = importlib.import_module(f"plugins.{plugin_name}")
                self.address_generators[plugin_name] = module.AddressGenerator()
                logging.info(f"Loaded address generator plugin: {plugin_name}")
            except ImportError as e:
                logging.error(f"Failed to load address generator plugin {plugin_name}: {e}")

    async def _connect_to_message_queue(self):
        try:
            url = urlparse(self.config.message_queue_url)
            parameters = pika.URLParameters(self.config.message_queue_url)
            connection = pika.BlockingConnection(parameters)
            self.message_queue_channel = connection.channel()
            self.message_queue_channel.queue_declare(queue='results')
            logging.info("Connected to message queue")
        except Exception as e:
            logging.error(f"Failed to connect to message queue: {e}")

    def _publish_to_message_queue(self, message: str):
        if self.message_queue_channel:
            try:
                self.message_queue_channel.basic_publish(exchange='', routing_key='results', body=message)
            except Exception as e:
                logging.error(f"Failed to publish to message queue: {e}")

    def _check_proxies(self):
        if not self.config.proxies:
            return

        logging.info("Checking proxies...")
        working_proxies = []
        for proxy in self.config.proxies:
            if proxy in self.dead_proxies:
                continue
            if self._is_proxy_working(proxy):
                working_proxies.append(proxy)
            else:
                self.dead_proxies.add(proxy)
                logging.warning(f"Proxy {proxy} is dead, removing it")

        self.config.proxies = working_proxies
        self.proxy_pool = cycle(self.config.proxies) if self.config.proxies else None
        if self.config.proxies:
            self.current_proxy = next(self.proxy_pool)
        else:
            self.current_proxy = None

        self.proxy_check_timer = threading.Timer(self.config.proxy_check_interval, self._check_proxies)
        self.proxy_check_timer.start()

    def _is_proxy_working(self, proxy: str) -> bool:
        try:
            if self.config.tor_proxy:
                socks_proxy = urlparse(self.config.tor_proxy)
                socks.set_default_proxy(socks.SOCKS5, socks_proxy.hostname, socks_proxy.port)
                socket.socket = socks.socksocket
            
            response = requests.get("http://www.google.com", proxies={"http": proxy, "https": proxy}, timeout=self.config.proxy_timeout)
            return response.status_code == 200
        except Exception:
            return False

    def _log_api_usage(self):
        with self.api_usage_lock:
            logging.info(f"API Usage: {self.api_usage}")
            self.api_usage = {"requests": 0, "errors": 0}
        self.api_usage_timer = threading.Timer(self.config.api_usage_log_interval, self._log_api_usage)
        self.api_usage_timer.start()

    def _is_valid_address(self, address: str) -> bool:
        try:
            # Basic address validation (can be improved)
            return len(address) >= 26 and len(address) <= 35 and (address.startswith('1') or address.startswith('3') or address.startswith('bc1'))
        except Exception:
            return False

    def _score_proxy(self, proxy: str, success: bool):
        if proxy not in self.proxy_scores:
            self.proxy_scores[proxy] = 0
        if success:
            self.proxy_scores[proxy] = min(self.proxy_scores[proxy] + 1, 10)
        else:
            self.proxy_scores[proxy] = max(self.proxy_scores[proxy] - 2, -10)

    def _get_best_proxy(self) -> str:
        if not self.config.proxies:
            return None
        
        valid_proxies = [p for p in self.config.proxies if self.proxy_scores.get(p, 0) >= self.config.proxy_score_threshold]
        if not valid_proxies:
            return None
        
        return random.choice(valid_proxies)

    def _update_tool(self):
        try:
            logging.info("Checking for updates...")
            with tempfile.TemporaryDirectory() as tmpdir:
                subprocess.check_call(['git', 'clone', '--depth', '1', self.config.update_url, tmpdir])
                
                # Copy necessary files (excluding virtual environment)
                for item in os.listdir(tmpdir):
                    s = os.path.join(tmpdir, item)
                    d = os.path.join(os.getcwd(), item)
                    if os.path.isdir(s) and item == '.git':
                        continue
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
            logging.info("Tool updated successfully. Please restart the tool.")
            os._exit(0)
        except Exception as e:
            logging.error(f"Failed to update tool: {e}")

    # Add the missing _update_stats method
    def _update_stats(self, balance: float, received: float):
        """Update statistics after checking an address"""
        with self.stats_lock:
            self.stats["checked"] += 1
            if balance > 0 or received > 0:
                self.stats["hits"] += 1

    # Add the missing async email notification method
    async def _send_email_notification_async(self, wallet: BTCWallet, address: str, balance: float, received: float):
        """Asynchronous version of email notification sender"""
        if not self.config.email_notifications:
            return
        
        # Use thread pool to run the blocking email operation
        await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            lambda: self._send_email_notification(wallet, address, balance, received)
        )

def main():
    try:
        miner = BTCMiner()

        def check_input():
            while True:
                user_input = input().lower()
                if user_input == 'p':
                    miner.pause()
                elif user_input == 'r':
                    miner.resume()
                elif user_input == 'q':
                    print("Exiting...")
                    os._exit(0)

        input_thread = threading.Thread(target=check_input, daemon=True)
        input_thread.start()

        asyncio.run(miner.main())
            
    except Exception as e:
        logging.exception("Fatal error")
        print(f"{Fore.RED}Error: {e}{Fore.RESET}")

if __name__ == "__main__":
    main()
