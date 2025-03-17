from Crypto.Hash import SHA3_256 as keccak_256  # Add this line
"""
Imports the SHA3_256 class from the 'Crypto.Hash' library with an alias 'keccak_256.'
This allows generating a keccak-256 hash from a given byte string. The code then
prints the hexadecimal digest of the hash for demonstration purposes.
"""

print(keccak_256(b'Hello').hexdigest())  # Use keccak_256 so it's accessed

# ...rest of the file...
