import sys
import importlib

def apply_patch():
    try:
        import hdwallet.hdwallet
        import hdwallet.utils
        import hdwallet.cryptocurrencies

        # Patch hdwallet to use pycryptodome for SHA-3 support
        hdwallet.hdwallet.sha3_256 = importlib.import_module('Crypto.Hash.SHA3_256').new
        hdwallet.utils.sha3_256 = importlib.import_module('Crypto.Hash.SHA3_256').new
        hdwallet.cryptocurrencies.sha3_256 = importlib.import_module('Crypto.Hash.SHA3_256').new

        print("hdwallet patched successfully to use pycryptodome for SHA-3 support.")
    except ImportError as e:
        print(f"Failed to apply hdwallet patch: {e}")

if __name__ == "__main__":
    apply_patch()
