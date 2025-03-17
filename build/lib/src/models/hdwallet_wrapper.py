from hdwallet import HDWallet as BaseHDWallet
from hdwallet.symbols import BTC
from Crypto.Hash import SHA3_256

class HDWallet(BaseHDWallet):
    def __init__(self, **kwargs):
        kwargs['symbol'] = BTC
        super().__init__(**kwargs)

    @staticmethod
    def _keccak256(data):
        if isinstance(data, str):
            data = data.encode()
        return SHA3_256.new(data).digest()
