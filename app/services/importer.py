"""Turns an uploaded statement (CSV, XLSX or PDF) into saved transactions.

The LLM works out each file's layout, so no bank needs its own parser.
"""
import csv
import datetime as dt
import hashlib
import io
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pdfplumber
from sqlmodel import Session, select

from app.db import clear_browser_data
from app.models import DEMO_SOURCE, UNCATEGORIZED, Transaction
from app.services.categorizer import categorize
from app.services.llm import ask_llm

MAX_FILE_BYTES = 10 * 1024 * 1024
SAMPLE_ROWS = 12
PAGES_PER_REQUEST = 3  # 5 pages took over a minute in one request, close to the timeout
PDF_WORKERS = 3  # requests at once: a 30-page statement took 77s this way, 212s one at a time
MAX_PDF_PAGES = 60  # 20 requests at most, which is the free limit for one minute

AMOUNT = re.compile(r"\d[\d,]*(?:\.\d+)?")
YEAR_FIRST_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

COLUMNS_SYSTEM = (
    "You read the layout of bank statement spreadsheets. You get the first rows, each with its "
    "row number and a list of cells. Find the header row of the transactions table and give "
    "0-based cell positions. Use debit_col and credit_col when withdrawals and deposits have "
    "separate columns. Otherwise use amount_col, plus type_col if a column says debit or credit. "
    "Use null for columns that do not exist. Set day_first to true if dates are written day "
    "before month, as is usual in India."
)
COLUMNS_SCHEMA = {
    "type": "object",
    "properties": {
        "header_row": {"type": "integer"},
        "date_col": {"type": "integer"},
        "description_col": {"type": "integer"},
        "debit_col": {"type": ["integer", "null"]},
        "credit_col": {"type": ["integer", "null"]},
        "amount_col": {"type": ["integer", "null"]},
        "type_col": {"type": ["integer", "null"]},
        "day_first": {"type": "boolean"},
    },
    "required": [
        "header_row",
        "date_col",
        "description_col",
        "debit_col",
        "credit_col",
        "amount_col",
        "type_col",
        "day_first",
    ],
    "additionalProperties": False,
}

PDF_SYSTEM = (
    "You extract transactions from the text of an Indian bank or UPI statement. Return every "
    "transaction exactly once, with the date as YYYY-MM-DD, the description as written, the "
    "amount as a positive number, and type 'expense' for money paid or debited, or 'income' for "
    "money received or credited. Do not include balances or totals, and do not invent rows. "
    "Numbers and IDs are masked as [NUM] and [ID]; leave them out of the description."
)
PDF_SCHEMA = {
    "type": "object",
    "properties": {
        "transactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "description": {"type": "string"},
                    "amount": {"type": "number"},
                    "type": {"type": "string", "enum": ["income", "expense"]},
                },
                "required": ["date", "description", "amount", "type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["transactions"],
    "additionalProperties": False,
}


class StatementError(Exception):
    """The file could not be read as a statement. The message is safe to show the user."""


def parse_amount(value) -> float | None:
    """Read a signed number from text such as '1,00,000.50', 'Rs. 350' or '-120'."""
    if value is None or pd.isna(value):
        return None
    text = str(value)
    match = AMOUNT.search(text)
    if not match:
        return None
    number = float(match.group().replace(",", ""))
    is_negative = text.strip().startswith(("-", "("))
    return -number if is_negative else number


def parse_date(text: str | None, day_first: bool) -> dt.date | None:
    """Read a date. Year-first dates, which Excel date cells become, must ignore day_first,
    or pandas would swap their day and month."""
    if text is None:
        return None
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=day_first and not YEAR_FIRST_DATE.match(text))
    return None if pd.isna(parsed) else parsed.date()


def _cell(row, position: int | None) -> str | None:
    """Read one cell as clean text, treating blanks as None."""
    if position is None or position not in row.index or pd.isna(row[position]):
        return None
    return str(row[position]).strip() or None


def _amount_and_type(row, columns: dict) -> tuple[float, str] | None:
    """Work out one row's amount and direction from the mapped columns."""
    if columns["debit_col"] is not None or columns["credit_col"] is not None:
        debit = parse_amount(_cell(row, columns["debit_col"]))
        credit = parse_amount(_cell(row, columns["credit_col"]))
        if debit:
            return abs(debit), "expense"
        if credit:
            return abs(credit), "income"
        return None

    raw = _cell(row, columns["amount_col"])
    amount = parse_amount(raw)
    if not amount:
        return None
    # A separate type column, or a suffix such as "1,200.00 Cr", gives the direction.
    flag = (_cell(row, columns["type_col"]) or raw).lower()
    if re.search(r"\b(cr|credit|credited|deposit|income|received)\b", flag):
        return abs(amount), "income"
    if re.search(r"\b(dr|debit|debited|withdrawal|expense|paid|sent)\b", flag):
        return abs(amount), "expense"
    return abs(amount), ("expense" if amount < 0 else "income")


def read_table(extension: str, data: bytes) -> pd.DataFrame:
    """Load a spreadsheet as plain text cells. Bank exports often have uneven rows above the table."""
    try:
        if extension == ".xlsx":
            return pd.read_excel(io.BytesIO(data), header=None, dtype=str)
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace"))))
        width = max((len(row) for row in rows), default=0)
        return pd.DataFrame([row + [None] * (width - len(row)) for row in rows])
    except Exception as exc:  # third-party parsers raise many types on a bad file
        raise StatementError("Could not read this file as a spreadsheet.") from exc


def sample_rows(table: pd.DataFrame) -> list[dict]:
    """Pick the few rows the LLM needs to see, because whole files are token heavy."""
    sample = []
    for index, row in table.head(60).iterrows():
        cells = [None if pd.isna(cell) or not str(cell).strip() else str(cell).strip()[:60] for cell in row]
        # Rows with one or two cells are usually name and address details, not table rows.
        if sum(cell is not None for cell in cells) >= 3:
            sample.append({"row": index, "cells": cells})
        if len(sample) == SAMPLE_ROWS:
            break
    return sample


def find_columns(table: pd.DataFrame) -> dict:
    """Ask the LLM for the spreadsheet's layout."""
    sample = sample_rows(table)
    if not sample:
        raise StatementError("Could not find a transactions table in this file.")
    return ask_llm(COLUMNS_SYSTEM, json.dumps(sample, ensure_ascii=False), COLUMNS_SCHEMA)


def rows_from_table(table: pd.DataFrame, columns: dict) -> tuple[list[dict], int]:
    """Read every row using the mapped columns. Returns rows, plus how many were skipped."""
    rows, skipped = [], 0
    header = columns["header_row"]
    # Start at the header itself, because the model is sometimes one row out. A header has no
    # date, so it cannot become a transaction, and this way the first real row is never lost.
    for index, row in table.iloc[header:].iterrows():
        if all(pd.isna(cell) or not str(cell).strip() for cell in row):
            continue
        date = parse_date(_cell(row, columns["date_col"]), columns["day_first"])
        money = _amount_and_type(row, columns)
        if date is None or money is None:  # the header, footers, totals and other non-transaction rows
            skipped += index != header
            continue
        rows.append(
            {
                "date": date,
                "description": _cell(row, columns["description_col"]) or "(no description)",
                "amount": round(money[0], 2),
                "type": money[1],
            }
        )
    return rows, skipped


def _amount_in_text(amount: float, text: str) -> bool:
    """Check an extracted amount really appears in the source, since models can invent rows."""
    plain = text.replace(",", "")
    return f"{amount:.2f}" in plain or (amount == int(amount) and str(int(amount)) in plain)


def rows_from_pdf(data: bytes) -> tuple[list[dict], int]:
    """Send the PDF's text to the LLM a few pages at a time. Returns rows, plus how many were skipped."""
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > MAX_PDF_PAGES:
                raise StatementError(
                    f"This PDF has {len(pdf.pages)} pages and the limit is {MAX_PDF_PAGES}. "
                    "Please download a shorter period."
                )
            pages = [page.extract_text() or "" for page in pdf.pages]
    except StatementError:
        raise
    except Exception as exc:  # third-party parsers raise many types on a bad file
        raise StatementError("Could not read this PDF. If it has a password, remove it first.") from exc
    if not any(page.strip() for page in pages):
        raise StatementError("This PDF has no readable text. It may be a scanned image.")

    texts = ["\n".join(pages[start : start + PAGES_PER_REQUEST]) for start in range(0, len(pages), PAGES_PER_REQUEST)]
    texts = [text for text in texts if text.strip()]

    # map() keeps page order. If one request fails, the import fails, but the others' answers are
    # saved by ask_llm, so trying again only resends the one that failed.
    with ThreadPoolExecutor(max_workers=PDF_WORKERS) as pool:
        answers = list(pool.map(lambda text: ask_llm(PDF_SYSTEM, text, PDF_SCHEMA), texts))

    rows, skipped = [], 0
    for text, answer in zip(texts, answers):
        for item in answer["transactions"]:
            try:
                date = dt.date.fromisoformat(item["date"])
                amount = abs(float(item["amount"]))
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if not amount or item.get("type") not in ("income", "expense") or not _amount_in_text(amount, text):
                skipped += 1
                continue
            rows.append(
                {
                    "date": date,
                    "description": str(item.get("description", "")).strip() or "(no description)",
                    "amount": round(amount, 2),
                    "type": item["type"],
                }
            )
    return rows, skipped


def _row_key(row: dict) -> str:
    description = " ".join(row["description"].lower().split())
    return f"{row['date']}|{description}|{row['amount']:.2f}|{row['type']}"


def _split_new_rows(rows: list[dict], source: str, session: Session) -> tuple[list[Transaction], int]:
    """Separate new rows from ones already imported. Returns new transactions and the duplicate count."""
    imported = session.exec(
        select(Transaction.dedupe_hash, Transaction.date, Transaction.amount, Transaction.type).where(
            Transaction.dedupe_hash.is_not(None)
        )
    ).all()
    payment_by_hash = {dedupe_hash: (date, amount, kind) for dedupe_hash, date, amount, kind in imported}
    unmatched = Counter(payment_by_hash.values())

    # Exact repeats, such as importing a file again. Counting repeats within the file keeps
    # two genuine identical rows, such as two teas in one day.
    seen = Counter()
    candidates, duplicates = [], 0
    for row in rows:
        key = _row_key(row)
        seen[key] += 1
        dedupe_hash = hashlib.sha256(f"{key}|{seen[key]}".encode()).hexdigest()
        payment = (row["date"], row["amount"], row["type"])
        if dedupe_hash in payment_by_hash:
            unmatched[payment] -= 1
            duplicates += 1
        else:
            candidates.append((row, dedupe_hash, payment))

    # The same payment can be in a bank statement and a UPI app statement with different
    # descriptions, so a matching date, amount and type counts as already imported.
    new = []
    for row, dedupe_hash, payment in candidates:
        if unmatched[payment] > 0:
            unmatched[payment] -= 1
            duplicates += 1
        else:
            new.append(Transaction(**row, source=source, dedupe_hash=dedupe_hash))
    return new, duplicates


def import_statement(filename: str, data: bytes, session: Session) -> dict:
    """Read a statement, skip rows already imported, categorize the new ones and save them."""
    extension = Path(filename or "").suffix.lower()
    if extension not in (".csv", ".xlsx", ".pdf"):
        raise StatementError("Please upload a .csv, .xlsx or .pdf file.")
    if len(data) > MAX_FILE_BYTES:
        raise StatementError("This file is larger than the 10 MB limit.")

    if extension == ".pdf":
        rows, skipped = rows_from_pdf(data)
    else:
        table = read_table(extension, data)
        rows, skipped = rows_from_table(table, find_columns(table))
    if not rows:
        raise StatementError("No transactions were found in this file.")

    # Real rows must not mix with made-up ones, or dedupe could drop them as duplicates of the demo.
    # This commits along with the new rows, so a failed upload leaves the demo data in place.
    replaced_demo = session.exec(select(Transaction.id).where(Transaction.source == DEMO_SOURCE)).first() is not None
    if replaced_demo:
        clear_browser_data(session)

    new, duplicates = _split_new_rows(rows, extension[1:], session)
    warning = categorize(new, session) if new else None
    session.add_all(new)
    session.commit()
    return {
        "imported": len(new),
        "duplicates": duplicates,
        "skipped": skipped,
        "uncategorized": sum(t.category == UNCATEGORIZED for t in new),
        "warning": warning,
        "replaced_demo_data": replaced_demo,
    }
