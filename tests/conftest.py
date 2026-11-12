"""Test setup: a temporary database and a fake LLM, so tests never use real requests."""
import json

import pytest
from fastapi.testclient import TestClient

from app import config, db
from app.main import app
from app.services import llm

# The layout of the CSV used in test_importer.py.
CSV_COLUMNS = {
    "header_row": 2,
    "date_col": 0,
    "description_col": 1,
    "debit_col": 2,
    "credit_col": 3,
    "amount_col": None,
    "type_col": None,
    "day_first": True,
}


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """All data goes to a temp folder, and a new browser starts empty unless a test says otherwise.
    Gives the shared database's engine."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    engine = db.sqlite_engine(tmp_path / "data" / "shared.db")
    monkeypatch.setattr(db, "shared_engine", engine)
    empty_seed = tmp_path / "empty_seed.db"
    db.create_tables(db.sqlite_engine(empty_seed), db.BROWSER_TABLES)
    monkeypatch.setattr(db, "SEED_PATH", empty_seed)
    db.init_db()
    return engine


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


class FakeLLM:
    """Stands in for OpenRouter and records what was sent."""

    def __init__(self):
        self.calls = []
        self.systems = []
        self.fail = False
        self.fail_categorizing = False
        self.columns = CSV_COLUMNS
        self.pdf_transactions = []
        self.insights = {"summary": "", "spending_cuts": [], "investment_advice": []}

    def __call__(self, system, user, schema, models):
        self.calls.append(user)
        self.systems.append(system)
        fields = schema["properties"]
        if self.fail or (self.fail_categorizing and "items" in fields):
            raise llm.LLMError("Rate limit reached.")
        if "items" in fields:
            answers = []
            for item in json.loads(user)["items"]:
                is_swiggy = "SWIGGY" in item["text"].upper()
                answers.append(
                    {
                        "id": item["id"],
                        "merchant": "Swiggy" if is_swiggy else "Some Shop",
                        "category": "Food & Dining" if is_swiggy else "Shopping",
                    }
                )
            return {"items": answers}, "fake-model"
        if "header_row" in fields:
            return dict(self.columns), "fake-model"
        if "spending_cuts" in fields:
            return dict(self.insights), "fake-model"
        return {"transactions": self.pdf_transactions}, "fake-model"


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "call_openrouter", fake)
    return fake
