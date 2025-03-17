import hashlib

class AddressGenerator:
    def generate_address(self, seed: str) -> str:
        return hashlib.sha3_256(seed.encode()).hexdigest()
