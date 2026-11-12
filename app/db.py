"""SQLite. Each browser gets its own database file, so visitors never see each other's statements.
One shared database keeps what is safe to reuse across them."""
import os
import shutil
import threading
from pathlib import Path

from fastapi import Request
from sqlalchemy import inspect
from sqlalchemy.pool import NullPool
from sqlmodel import Session, SQLModel, create_engine, delete

from app import config
from app.models import (
    Budget,
    DescriptionAnswer,
    AuthSession,
    FundSnapshot,
    InsightReport,
    LLMCache,
    MerchantCategory,
    SearchCache,
    Transaction,
    User,
)

# What a new browser starts with: made-up demo data. Build it with scripts/make_seed_db.py.
SEED_PATH = Path(__file__).resolve().parent.parent / "samples" / "seed.db"

# Saved LLM answers and fund prices save requests for everyone. Never send these tables to a visitor:
# DescriptionAnswer holds masked statement text, and a person's name can survive masking.
# User and AuthSession (the POC login) live here too, and no endpoint returns their rows.
SHARED_TABLES = (LLMCache, DescriptionAnswer, FundSnapshot, SearchCache, User, AuthSession)
# MerchantCategory is private because it holds a user's corrections and the names sent to the LLM.
BROWSER_TABLES = (Transaction, MerchantCategory, Budget, InsightReport)

# The first requests from a new page arrive together, and only one of them may copy the seed.
_seed_lock = threading.Lock()


def sqlite_engine(path: Path):
    """No connection pool: a connection closes with its session, so no browser's file stays open."""
    return create_engine(
        f"sqlite:///{path.as_posix()}", connect_args={"check_same_thread": False}, poolclass=NullPool
    )


shared_engine = sqlite_engine(config.DATA_DIR / "shared.db")


def create_tables(engine, models) -> None:
    SQLModel.metadata.create_all(engine, tables=[model.__table__ for model in models])


def init_db() -> None:
    """Create the data folder and any shared tables that do not exist yet."""
    (config.DATA_DIR / "browsers").mkdir(parents=True, exist_ok=True)
    create_tables(shared_engine, SHARED_TABLES)
    _add_user_name_column()


def _add_user_name_column() -> None:
    """The POC login first shipped without User.name, and create_all never alters a table that
    exists. So a shared.db made by that version gets the column here."""
    if "name" not in {c["name"] for c in inspect(shared_engine).get_columns("user")}:
        with shared_engine.begin() as connection:
            connection.exec_driver_sql('ALTER TABLE "user" ADD COLUMN name VARCHAR NOT NULL DEFAULT \'\'')


def _browser_path(browser_id: str) -> Path:
    return config.DATA_DIR / "browsers" / f"{browser_id}.db"


def _copy_seed(path: Path) -> None:
    temp = path.with_suffix(".tmp")
    shutil.copyfile(SEED_PATH, temp)
    os.replace(temp, path)  # all at once, so no request can open half a copy


def browser_engine(browser_id: str):
    """This browser's database, starting from the demo data on its first use."""
    path = _browser_path(browser_id)
    with _seed_lock:
        if not path.exists():
            _copy_seed(path)
    return sqlite_engine(path)


def get_session(request: Request):
    """Give each API request a session on its own browser's database."""
    with Session(browser_engine(request.state.browser_id)) as session:
        yield session


def clear_browser_data(session: Session) -> None:
    """Empty a browser's database. The caller commits, so a failed request changes nothing."""
    for model in BROWSER_TABLES:
        session.exec(delete(model))


def reset_browser_data(browser_id: str, demo: bool) -> None:
    """Start this browser again, with the demo data or with nothing."""
    path = _browser_path(browser_id)
    with _seed_lock:
        _copy_seed(path)
    if not demo:
        with Session(sqlite_engine(path)) as session:
            clear_browser_data(session)
            session.commit()
