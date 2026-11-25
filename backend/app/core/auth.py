"""JWT authentication, bcrypt password hashing, and user persistence."""

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.bcrypt import BcryptHasher
from sqlalchemy import DateTime, Engine, Integer, String, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from ..config import settings
from .database import get_engine


class AuthBase(DeclarativeBase):
    """Declarative base for authentication tables."""


class User(AuthBase):
    """Authenticated QueryPilot user account."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )


_password_hasher = PasswordHash((BcryptHasher(),))
_bearer_scheme = HTTPBearer(auto_error=False)


def ensure_users_table(target_engine: Engine | None = None) -> None:
    """Create the users table if it does not yet exist."""
    eng = target_engine or get_engine()
    AuthBase.metadata.create_all(bind=eng)


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return _password_hasher.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""
    return _password_hasher.verify(plain_password, hashed_password)


def _require_jwt_secret() -> str:
    """Return configured JWT_SECRET_KEY or fail loudly if missing."""
    secret = settings.jwt_secret_key.strip()
    if not secret:
        raise RuntimeError("JWT_SECRET_KEY is not configured.")
    return secret


def create_access_token(user_id: int, email: str) -> tuple[str, int]:
    """Create a signed HS256 JWT access token and return (token, ttl_seconds)."""
    secret = _require_jwt_secret()
    ttl_seconds = settings.jwt_expire_minutes * 60
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_expire_minutes)).timestamp()),
    }
    token = jwt.encode(payload, secret, algorithm=settings.jwt_algorithm)
    return token, ttl_seconds


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a Bearer JWT access token."""
    secret = _require_jwt_secret()
    try:
        return jwt.decode(token, secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def register_user(email: str, password: str, target_engine: Engine | None = None) -> User:
    """Register a new user account with a bcrypt-hashed password."""
    eng = target_engine or get_engine()
    ensure_users_table(eng)
    normalized_email = email.strip().lower()

    with Session(eng) as session:
        existing = session.scalar(select(User).where(User.email == normalized_email))
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists.",
            )

        user = User(
            email=normalized_email,
            hashed_password=hash_password(password),
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def authenticate_user(email: str, password: str, target_engine: Engine | None = None) -> User:
    """Verify user credentials and return the User record."""
    eng = target_engine or get_engine()
    ensure_users_table(eng)
    normalized_email = email.strip().lower()

    with Session(eng) as session:
        user = session.scalar(select(User).where(User.email == normalized_email))
        if user is None or not verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> User:
    """FastAPI dependency enforcing Bearer JWT authentication."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    subject = payload.get("sub")
    if subject is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = int(subject)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed subject claim in token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    eng = get_engine()
    ensure_users_table(eng)
    with Session(eng) as session:
        user = session.scalar(select(User).where(User.id == user_id))
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authenticated user no longer exists.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
