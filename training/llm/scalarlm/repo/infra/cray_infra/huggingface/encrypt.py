from cryptography.fernet import Fernet

key = b"JAJOZunNSRFeXWXWVVVJfiKSzdzFMw0yFn8_JK50h60="

cipher = Fernet(key)

plaintext = ""
encrypted = cipher.encrypt(plaintext.encode())

print(f"Encrypted: {encrypted}")
# Paste the output into Config.hf_encrypted_token

# Sanity check — decrypt it back
decrypted = cipher.decrypt(encrypted).decode()
print(f"Decrypted: {decrypted}")
assert decrypted == plaintext, "Round-trip failed!"
