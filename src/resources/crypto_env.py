import os
import base64
import re
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ENC(...) 패턴
ENC_PATTERN = re.compile(r"^ENC\((.+)\)$")


# =========================================================
# Key Loader
# =========================================================
def _get_key() -> bytes:
    key = os.getenv("APP_SECRET_KEY")

    if not key:
        raise RuntimeError("APP_SECRET_KEY environment variable not set")

    key_bytes = key.encode()

    if len(key_bytes) != 32:
        raise RuntimeError("APP_SECRET_KEY must be exactly 32 bytes")

    return key_bytes


# =========================================================
# Encrypt → ENC(...)
# =========================================================
def encrypt_secret(plain: str) -> str:
    """
    plaintext → ENC(...)
    """
    key = _get_key()
    aes = AESGCM(key)

    nonce = os.urandom(12)

    ciphertext = aes.encrypt(
        nonce,
        plain.encode(),
        None,
    )

    blob = nonce + ciphertext
    b64 = base64.urlsafe_b64encode(blob).decode()

    return f"ENC({b64})"


# =========================================================
# Decrypt ENC(...)
# =========================================================
def decrypt_secret(value: str) -> str:
    """
    ENC(...) → plaintext
    plaintext → 그대로 반환
    """
    if not isinstance(value, str):
        return value

    match = ENC_PATTERN.match(value)
    if not match:
        return value

    blob = base64.urlsafe_b64decode(match.group(1))

    nonce = blob[:12]
    ciphertext = blob[12:]

    key = _get_key()
    aes = AESGCM(key)

    plain = aes.decrypt(
        nonce,
        ciphertext,
        None,
    )

    return plain.decode()


# =========================================================
# ENV helper (가장 많이 쓰게 됨)
# =========================================================
def get_secret(env_name: str, default=None):
    """
    ENV 읽고 자동 decrypt
    """
    value = os.getenv(env_name, default)
    return decrypt_secret(value)