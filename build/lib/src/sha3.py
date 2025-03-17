"""
Mock sha3 module that uses pycryptodome's SHA3_256
"""
from Crypto.Hash import SHA3_256 as _SHA3_256

class Keccak256:
    def __init__(self, data=None):
        self.hash_obj = _SHA3_256.new()
        if data is not None:
            if isinstance(data, str):
                data = data.encode('utf-8')
            self.hash_obj.update(data)
            
    def update(self, data):
        if isinstance(data, str):
            data = data.encode('utf-8')
        self.hash_obj.update(data)
        
    def hexdigest(self):
        return self.hash_obj.hexdigest()
        
    def digest(self):
        return self.hash_obj.digest()

def keccak_256(data=None):
    return Keccak256(data)

# Export the function directly so it can be imported as `from sha3 import keccak_256`
__all__ = ['keccak_256']
