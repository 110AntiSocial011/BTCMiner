# Bitcoin Wallets Miner
Bitcoin seeds bruteforcer / checker | First legit tool

For issues or improvements contact **@LizardX2** on Telegram.

## Installation

### Quick Install
```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Manual Installation
If you prefer installing packages individually:
```bash
pip install colorama hdwallet requests aioredis aiohttp aioh2 aiosqlite prometheus_client \
    psutil websockets ujson pika PySocks memory_profiler tqdm ratelimit backoff click pandas pycryptodome
```

## Setup

1. Make sure the required files are in the project path:
   - `config.json` - Main configuration file (see example below)
   - `check.txt` - File containing seeds to check (for Checker mode)
   - `failed.txt` - Output file for wallets with no balance (created automatically)
   - `success.txt` - Output file for wallets with balance (created automatically)

2. Customize `config.json` according to your requirements (see Configuration section)

## Running the Application

### From Command Line
```bash
# Run the main application
python btcminer.py

# When prompted, choose a mode:
# - B: Bruteforcer mode (generate and check seeds)
# - C: Checker mode (check seeds from file)
# - T: Test mode (validate configuration)
# - E: Export mode (export results)
```

### Available Commands While Running
- `p`: Pause the current operation
- `r`: Resume a paused operation
- `q`: Quit the application

## Testing
```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src

# Run specific test file
pytest tests/test_wallet.py
```

## Configuration

The `config.json` file controls all aspects of the application:

```json
{
    "settings": {
        "checker": {
            "filename": "check.txt"  // File with seeds to check
        },
        "bruteforcer": {
            "strength": 128,  // Seed strength (128, 160, 192, 224, or 256)
            "language": "english",  // Mnemonic language
            "passphrase": "None"  // Optional passphrase
        },
        "general": { 
            "failed": "failed.txt",  // Output for failed attempts
            "success": "success.txt",  // Output for successful finds
            "addresstype" : "p2pkh",  // Address type to generate
            "api": {
                "api_url": "https://chain.api.btc.com/v3/address",  // API URL
                "api_get_data": "data",  // JSON path to data
                "api_get_balance": "balance",  // JSON path to balance
                "api_get_received": "received"  // JSON path to received amount
            }
        }
    }
}
```

## How It Works

**Bruteforce Mode**: Generates seed → Retrieves private key, address, wif and more → checks balance → stores on the chosen files.

**Checker Mode**: Takes seeds from the chosen file → Retrieves private key, address, wif and more → checks balance → stores on the chosen files.

## Advanced Usage

### Using Redis Cache
Configure Redis URL in `config.json` to enable caching of balance checks:
```json
"redis_url": "redis://localhost"
```

### Monitoring with Prometheus
The application exposes metrics on port 9090 by default. You can use Prometheus to scrape these metrics.

### Using Proxies
Add proxies to `config.json` to distribute API requests:
```json
"proxies": ["http://proxy1.example.com", "http://proxy2.example.com"]
```

## Screenshots

![Tool Screenshot](https://user-images.githubusercontent.com/108220470/185769833-ef1a9191-6e41-4fcb-b300-d59ee372b276.png)

![Config Screenshot](https://user-images.githubusercontent.com/108220470/185769906-77f04412-a36d-4eab-a013-57675925c15e.png)
