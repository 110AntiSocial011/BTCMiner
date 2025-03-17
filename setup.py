from setuptools import setup, find_packages

setup(
    name="btcminer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'colorama',
        'hdwallet',
        'requests',
        'aioredis',
        'aiohttp',
        'aioh2',
        'aiosqlite',
        'prometheus_client',
        'psutil',
        'websockets',
        'ujson',
        'pika',
        'PySocks',
        'memory_profiler',
        'tqdm',
        'ratelimit',
        'backoff',
        'click',
        'pycryptodome>=3.10.1',  # Use pycryptodome for SHA-3 support
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-asyncio',
            'pytest-mock',
            'pytest-cov',
        ],
    },
)
