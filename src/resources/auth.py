from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException

from src.config import get_jwt_secret_key, settings


def create_access_token(*, emp_id: str, email: str) -> str:
    now = datetime.now(timezone.utc)
    expire_at = now + timedelta(minutes=settings.jwt_expire_min)
    payload = {
        "sub": email,
        "emp_id": emp_id,
        "iat": int(now.timestamp()),
        "exp": int(expire_at.timestamp()),
    }
    return jwt.encode(
        payload,
        get_jwt_secret_key(),
        algorithm=settings.jwt_algorithm,
    )


def verify_access_token(token: str) -> dict[str, object]:
    try:
        decoded = jwt.decode(
            token,
            get_jwt_secret_key(),
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc
    return decoded
