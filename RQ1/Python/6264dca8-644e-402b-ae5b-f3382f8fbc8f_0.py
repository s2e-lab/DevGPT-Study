import hashlib

# Input data as a string
data = "Hello, SHA-256!"

# Convert the input string to bytes (UTF-8 encoding)
data_bytes = data.encode('utf-8')

# Create a SHA-256 hash object
sha256_hash = hashlib.sha256()

# Update the hash object with the input bytes
sha256_hash.update(data_bytes)

# Get the hexadecimal representation of the hash
hash_hex = sha256_hash.hexdigest()

# Print the SHA-256 hash
print("SHA-256 Hash:", hash_hex)
