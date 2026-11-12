"""A proof-of-concept login: email + password, and a cookie that names the logged-in user.

A logged-in request uses the user's own database file instead of the browser's, so every router
works unchanged. Not production grade: no rate limiting, no email check, no password reset.
"""
import datetime as dt
import hashlib
import hmac
import secrets

from sqlmodel import Session, select

from app import db
from app.models import AuthSession, User

SESSION_DAYS = 7


class AuthError(Exception):
    """The message is safe to show the user."""


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"{salt.hex()}${digest.hex()}"


def check_password(password: str, stored: str) -> bool:
    salt_hex, _, _ = stored.partition("$")
    return hmac.compare_digest(hash_password(password, bytes.fromhex(salt_hex)), stored)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _start_session(session: Session, user: User) -> str:
    """Returns the token for the cookie."""
    token = secrets.token_urlsafe(32)
    session.add(AuthSession(token_hash=_token_hash(token), user_id=user.id))
    session.commit()
    return token


def signup(email: str, password: str, name: str = "") -> tuple[User, str]:
    email = email.strip().lower()
    with Session(db.shared_engine, expire_on_commit=False) as session:
        if session.exec(select(User).where(User.email == email)).first():
            raise AuthError("That email is already registered.")
        user = User(email=email, name=name, password_hash=hash_password(password), data_key=secrets.token_hex(16))
        session.add(user)
        session.commit()
        session.refresh(user)
        return user, _start_session(session, user)


def login(email: str, password: str) -> tuple[User, str]:
    with Session(db.shared_engine, expire_on_commit=False) as session:
        user = session.exec(select(User).where(User.email == email.strip().lower())).first()
        # One message for both cases, so the form does not reveal which emails are registered
        if not user or not check_password(password, user.password_hash):
            raise AuthError("Wrong email or password.")
        return user, _start_session(session, user)


def user_for_token(token: str) -> User | None:
    """The user a session cookie belongs to, or None if it is unknown or older than SESSION_DAYS."""
    if not token:
        return None
    with Session(db.shared_engine, expire_on_commit=False) as session:
        found = session.get(AuthSession, _token_hash(token))
        if not found:
            return None
        created = found.created_at.replace(tzinfo=dt.timezone.utc)  # SQLite drops the timezone
        if dt.datetime.now(dt.timezone.utc) - created > dt.timedelta(days=SESSION_DAYS):
            session.delete(found)
            session.commit()
            return None
        return session.get(User, found.user_id)


def logout(token: str) -> None:
    with Session(db.shared_engine, expire_on_commit=False) as session:
        found = session.get(AuthSession, _token_hash(token))
        if found:
            session.delete(found)
            session.commit()
