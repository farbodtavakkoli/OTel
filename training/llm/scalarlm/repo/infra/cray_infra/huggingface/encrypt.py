from cryptography.fernet import Fernet

key = b"JAJOZunNSRFeXWXWVVVJfiKSzdzFMw0yFn8_JK50h60="

cipher = Fernet(key)

plaintext = ""
encrypted = cipher.encrypt(plaintext.encode())

print(f"Encrypted: {encrypted}")

decrypted = cipher.decrypt(encrypted).decode()
print(f"Decrypted: {decrypted}")
assert decrypted == plaintext, "Round-trip failed!"
