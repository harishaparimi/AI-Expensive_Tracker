"""Database tables and API request shapes."""
import datetime as dt
from typing import Annotated, Literal

from fastapi import Query
from pydantic import StringConstraints
from sqlmodel import Field, SQLModel

CATEGORIES = (
    "Food & Dining",
    "Groceries",
    "Transport",
    "Shopping",
    "Bills & Utilities",
    "Rent & Housing",
    "Entertainment",
    "Subscriptions",
    "Health",
    "Education",
    "Travel",
    "Investments",
    "Transfers",
    "Income",
    "Other",
)
UNCATEGORIZED = "Uncategorized"
UNKNOWN_MERCHANT = "Unknown"
DEMO_SOURCE = "demo"  # the source of the made-up rows a new browser starts with

# Moving your own money is not spending, so a SIP or a friend's repayment must not look like it.
NOT_SPENDING = ("Transfers", "Investments")
SPENDING_CATEGORIES = tuple(c for c in CATEGORIES if c not in (*NOT_SPENDING, "Income"))
# Money coming in under one of these is a refund. "Other" is left out, because a cash deposit
# is not a refund, and "Uncategorized" too, because a salary is that until the LLM has answered.
REFUND_CATEGORIES = tuple(c for c in SPENDING_CATEGORIES if c != "Other")
# Never advised as a spending cut: needs, and "Other", which is cash withdrawals and anything unnamed.
NOT_TO_CUT = ("Rent & Housing", "Bills & Utilities", "Health", "Other")

Category = Literal[CATEGORIES]
BudgetCategory = Literal[SPENDING_CATEGORIES]
TransactionType = Literal["income", "expense"]
# An optional ?month=2026-10 filter. Left out, it means the latest month with data.
MonthParam = Annotated[str | None, Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]


class Transaction(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    date: dt.date = Field(index=True)
    description: str
    merchant: str = Field(default=UNKNOWN_MERCHANT, index=True)
    amount: float  # always positive; `type` gives the direction
    type: str  # "income" or "expense"
    category: str = Field(default=UNCATEGORIZED, index=True)
    category_source: str = "none"  # "llm", "user" or "none"
    source: str = "manual"  # "manual", "csv", "xlsx", "pdf" or DEMO_SOURCE
    dedupe_hash: str | None = Field(default=None, unique=True)  # imported rows only


class MerchantCategory(SQLModel, table=True):
    """Remembers each merchant's spelling and any user correction, which applies to future rows."""

    key: str = Field(primary_key=True)  # lowercase merchant name
    name: str
    category: str
    source: str  # "llm" or "user"


class DescriptionAnswer(SQLModel, table=True):
    """The LLM's answer for one masked description, so it is only ever asked once.
    A user's correction in MerchantCategory still wins over this."""

    text: str = Field(primary_key=True)
    type: str = Field(primary_key=True)  # a refund can read like a payment, so direction counts
    merchant: str
    category: str


class Budget(SQLModel, table=True):
    category: str = Field(primary_key=True)
    monthly_limit: float


class LLMCache(SQLModel, table=True):
    """Saved LLM answers, so a repeated prompt uses no request."""

    prompt_hash: str = Field(primary_key=True)
    response: str
    model: str
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class FundSnapshot(SQLModel, table=True):
    """The last fund figures fetched, reused for a day and if the market API is down."""

    code: int = Field(primary_key=True)  # mfapi.in scheme code
    fetched_on: dt.date
    snapshot: str  # JSON


class SearchCache(SQLModel, table=True):
    """Web search results, reused for a day. A repeated question then sends no search, and a search
    outage does not stop the chat. Queries are generic, so this is safe to share."""

    query: str = Field(primary_key=True)
    fetched_on: dt.date
    results: str  # JSON


class User(SQLModel, table=True):
    """A signed-up person. `data_key` names their own database file, so their data follows them
    to any browser."""

    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)  # lowercase
    name: str = ""  # shown on the sign-in page only
    password_hash: str  # "salt$hash", both hex
    data_key: str = Field(unique=True)  # random 32 hex, the same shape as a browser id
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class AuthSession(SQLModel, table=True):
    """A login. Only the hash of the cookie's token is kept, so a copy of this table cannot be used."""

    token_hash: str = Field(primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class InsightReport(SQLModel, table=True):
    """Generated reports. The newest is shown on page load, using no LLM request."""

    id: int | None = Field(default=None, primary_key=True)
    report: str  # JSON


class LoginRequest(SQLModel):
    email: Annotated[str, StringConstraints(max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]
    password: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    remember: bool = True  # False: the cookie ends when the browser closes


class SignupRequest(LoginRequest):
    password: Annotated[str, StringConstraints(min_length=8, max_length=200)]  # stricter than login
    name: Annotated[str, StringConstraints(strip_whitespace=True, max_length=80)] = ""


class BudgetUpdate(SQLModel):
    monthly_limit: float = Field(gt=0)


class TransactionCreate(SQLModel):
    date: dt.date
    description: str = Field(min_length=1)
    amount: float = Field(gt=0)
    type: TransactionType
    category: Category | None = None  # left out: the LLM picks one


class TransactionRestore(SQLModel):
    """A deleted row sent back as GET returned it, so Undo keeps what a new row could not have."""

    date: dt.date
    description: str = Field(min_length=1)
    merchant: str
    amount: float = Field(gt=0)
    type: TransactionType
    category: str
    category_source: str
    source: str
    dedupe_hash: str | None = None


class ChatTurn(SQLModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=4000)


class ChatRequest(SQLModel):
    """The page keeps the conversation in memory and sends it back with each question."""

    message: str = Field(min_length=1, max_length=500)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class TransactionUpdate(SQLModel):
    date: dt.date | None = None
    description: str | None = Field(default=None, min_length=1)
    amount: float | None = Field(default=None, gt=0)
    type: TransactionType | None = None
    category: Category | None = None
