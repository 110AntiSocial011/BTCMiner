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

# Provide the function directly
def keccak_256(data=None):
    hasher = Keccak256(data)
    return hasher

# This is necessary to match the sha3 module API
keccak_256.new = Keccak256  

__all__ = ['keccak_256']
