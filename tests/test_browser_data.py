"""Each browser has its own data, and shares only saved LLM answers and fund prices with the others."""
import datetime as dt
import json
import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect
from sqlmodel import Session, select

from app import config, db
from app.main import BROWSER_COOKIE, app
from app.models import (
    DEMO_SOURCE,
    UNCATEGORIZED,
    Budget,
    InsightReport,
    MerchantCategory,
    Transaction,
)

COMMITTED_SEED = db.SEED_PATH  # read here, before a fixture points it at a temp file
SWIGGY = {"date": "2026-10-03", "description": "SWIGGY order", "amount": 250, "type": "expense"}
# The layout matches CSV_COLUMNS in conftest. Its one row has the demo row's date, amount and type.
STATEMENT = (
    "MY BANK\n"
    "Account holder:,Someone\n"
    "Date,Narration,Debit,Credit,Balance\n"
    "01/10/26,UPI-SWIGGY-612345678901-PAYMENT,100.00,,5000.00\n"
).encode()


def new_browser() -> TestClient:
    return TestClient(app)


def browser_files() -> list:
    return list((config.DATA_DIR / "browsers").glob("*.db"))


def upload(browser, data=STATEMENT, name="statement.csv"):
    return browser.post("/import", files={"file": (name, data)})


@pytest.fixture
def demo_seed(tmp_path, monkeypatch):
    path = tmp_path / "demo_seed.db"
    engine = db.sqlite_engine(path)
    db.create_tables(engine, db.BROWSER_TABLES)
    with Session(engine) as session:
        session.add(
            Transaction(
                date=dt.date(2026, 10, 1),
                description="DEMO SHOP",
                merchant="Demo Shop",
                amount=100,
                type="expense",
                category="Shopping",
                category_source="llm",
                source=DEMO_SOURCE,
                dedupe_hash="demo-row",
            )
        )
        session.add(Budget(category="Shopping", monthly_limit=5000))
        session.add(MerchantCategory(key="demo shop", name="Demo Shop", category="Shopping", source="llm"))
        session.add(InsightReport(report=json.dumps({"summary": "Saved demo report"})))
        session.commit()
    monkeypatch.setattr(db, "SEED_PATH", path)


def test_two_browsers_do_not_see_each_others_data(fake_llm):
    mine, other = new_browser(), new_browser()
    row = mine.post("/transactions", json=SWIGGY).json()
    mine.put("/budgets/Shopping", json={"monthly_limit": 5000})

    assert len(mine.get("/transactions").json()) == 1
    assert other.get("/transactions").json() == []
    assert other.get(f"/transactions/{row['id']}").status_code == 404
    assert other.get("/budgets").json()["budgets"] == []
    assert mine.cookies[BROWSER_COOKIE] != other.cookies[BROWSER_COOKIE]
    assert len(browser_files()) == 2


def test_a_report_stays_with_its_browser(demo_seed):
    mine, other = new_browser(), new_browser()
    assert mine.get("/insights").json()["summary"] == "Saved demo report"
    mine.post("/reset", params={"demo": False})
    assert mine.get("/insights").status_code == 404
    assert other.get("/insights").status_code == 200


def test_the_same_browser_keeps_its_data_across_requests(fake_llm):
    browser = new_browser()
    browser.post("/transactions", json=SWIGGY)
    cookie = browser.cookies[BROWSER_COOKIE]

    returning = new_browser()
    returning.cookies.set(BROWSER_COOKIE, cookie)
    assert len(returning.get("/transactions").json()) == 1
    assert len(browser_files()) == 1


def test_the_cookie_cannot_be_read_by_scripts_or_sent_by_other_sites():
    cookie = new_browser().get("/categories").headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert "secure" not in cookie  # plain http here, where a secure cookie would never come back


def test_the_cookie_is_secure_over_https():
    browser = TestClient(app, base_url="https://testserver")
    assert "secure" in browser.get("/categories").headers["set-cookie"].lower()


def test_a_returning_browser_is_not_given_a_new_cookie():
    browser = new_browser()
    browser.get("/categories")
    assert "set-cookie" not in browser.get("/categories").headers


def test_a_page_load_alone_creates_no_database():
    new_browser().get("/categories")
    assert browser_files() == []


@pytest.mark.parametrize("bad", ["../../evil", "..%2F..%2Fevil", "abc", "A" * 32, "g" * 32, ""])
def test_a_cookie_in_any_other_form_is_replaced_and_never_used_as_a_path(tmp_path, bad):
    browser = new_browser()
    browser.cookies.set(BROWSER_COOKIE, bad)
    response = browser.get("/transactions")
    assert response.status_code == 200
    assert BROWSER_COOKIE in response.headers["set-cookie"]

    created = [path for path in tmp_path.rglob("*") if path.is_file() and path.suffix == ".db"]
    assert len(browser_files()) == 1
    assert sorted(path.name for path in created if path.parent.name != "browsers") == ["empty_seed.db", "shared.db"]


def test_requests_arriving_together_share_one_copy_of_the_seed(demo_seed):
    browser = new_browser()
    browser.get("/categories")  # the page itself, which sets the cookie
    cookie = browser.cookies[BROWSER_COOKIE]

    counts = []

    def read():
        one = TestClient(app)
        one.cookies.set(BROWSER_COOKIE, cookie)
        counts.append(len(one.get("/transactions").json()))

    threads = [threading.Thread(target=read) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert counts == [1] * 8
    assert len(browser_files()) == 1


def test_a_new_browser_starts_with_its_own_copy_of_the_demo_data(demo_seed):
    mine, other = new_browser(), new_browser()
    row = mine.get("/transactions").json()[0]
    assert row["source"] == DEMO_SOURCE
    assert mine.get("/budgets").json()["budgets"][0]["category"] == "Shopping"

    mine.delete(f"/transactions/{row['id']}")
    assert mine.get("/transactions").json() == []
    assert len(other.get("/transactions").json()) == 1


def test_reset_restores_the_demo_data(demo_seed, fake_llm):
    browser = new_browser()
    row = browser.get("/transactions").json()[0]
    browser.delete(f"/transactions/{row['id']}")
    browser.post("/transactions", json=SWIGGY)
    browser.delete("/budgets/Shopping")

    assert browser.post("/reset").status_code == 204

    assert [t["description"] for t in browser.get("/transactions").json()] == ["DEMO SHOP"]
    assert browser.get("/budgets").json()["budgets"][0]["category"] == "Shopping"
    assert browser.get("/insights").status_code == 200


def test_reset_can_start_empty(demo_seed):
    browser = new_browser()
    assert browser.post("/reset", params={"demo": False}).status_code == 204

    assert browser.get("/transactions").json() == []
    assert browser.get("/budgets").json()["budgets"] == []
    assert browser.get("/insights").status_code == 404
    assert browser.get("/analytics/trend").json() == []


def test_a_reset_only_touches_its_own_browser(demo_seed, fake_llm):
    mine, other = new_browser(), new_browser()
    other.post("/transactions", json=SWIGGY)
    mine.post("/reset", params={"demo": False})
    assert len(other.get("/transactions").json()) == 2


def test_an_upload_replaces_the_demo_data(demo_seed, fake_llm):
    browser = new_browser()
    result = upload(browser).json()

    assert result["replaced_demo_data"] is True
    assert [t["source"] for t in browser.get("/transactions").json()] == ["csv"]
    assert browser.get("/budgets").json()["budgets"] == []
    assert browser.get("/insights").status_code == 404


def test_a_real_row_is_not_dropped_as_a_duplicate_of_a_demo_row(demo_seed, fake_llm):
    assert upload(new_browser()).json()["imported"] == 1
    assert upload(new_browser()).json()["duplicates"] == 0


def test_demo_merchant_names_are_not_sent_with_a_real_upload(demo_seed, fake_llm):
    upload(new_browser())
    categorizing = [json.loads(call) for call in fake_llm.calls if '"known_merchants"' in call]
    assert categorizing[0]["known_merchants"] == []


def test_later_uploads_keep_what_is_already_there(demo_seed, fake_llm):
    browser = new_browser()
    upload(browser)
    browser.put("/budgets/Shopping", json={"monthly_limit": 7000})
    second = (
        "Date,Narration,Debit,Credit,Balance\n"
        "02/10/26,UPI-SOME SHOP-612345678902-PAYMENT,200.00,,4800.00\n"
    ).encode()
    fake_llm.columns = {**fake_llm.columns, "header_row": 0}
    result = upload(browser, second).json()

    assert result["replaced_demo_data"] is False
    assert len(browser.get("/transactions").json()) == 2
    assert browser.get("/budgets").json()["budgets"][0]["monthly_limit"] == 7000


def test_a_file_that_cannot_be_read_leaves_the_demo_data_in_place(demo_seed, fake_llm):
    browser = new_browser()
    assert upload(browser, b"not a statement", "statement.txt").status_code == 400
    assert browser.get("/transactions").json()[0]["source"] == DEMO_SOURCE


def test_an_llm_failure_on_the_layout_leaves_the_demo_data_in_place(demo_seed, fake_llm):
    browser = new_browser()
    fake_llm.fail = True
    assert upload(browser).status_code == 502
    assert browser.get("/transactions").json()[0]["source"] == DEMO_SOURCE
    assert browser.get("/insights").status_code == 200


def test_a_description_answered_for_one_browser_needs_no_request_for_another(fake_llm):
    new_browser().post("/transactions", json=SWIGGY)
    assert len(fake_llm.calls) == 1

    other = new_browser()
    assert other.post("/transactions", json=SWIGGY).json()["category"] == "Food & Dining"
    assert len(fake_llm.calls) == 1


def test_a_users_correction_does_not_reach_another_browser(fake_llm):
    mine = new_browser()
    row = mine.post("/transactions", json=SWIGGY).json()
    assert mine.patch(f"/transactions/{row['id']}", json={"category": "Shopping"}).status_code == 200

    other = new_browser()
    assert other.post("/transactions", json=SWIGGY).json()["category"] == "Food & Dining"
    assert mine.post("/transactions", json=SWIGGY).json()["category"] == "Shopping"


def test_merchant_names_sent_to_the_llm_come_only_from_this_browser(fake_llm):
    new_browser().post("/transactions", json=SWIGGY)
    new_browser().post("/transactions", json={**SWIGGY, "description": "SOME SHOP"})
    assert json.loads(fake_llm.calls[-1])["known_merchants"] == []


def test_each_database_holds_only_its_own_tables(client, temp_db):
    client.get("/transactions")
    shared = set(inspect(temp_db).get_table_names())
    assert shared == {model.__tablename__ for model in db.SHARED_TABLES}

    private = set(inspect(db.sqlite_engine(browser_files()[0])).get_table_names())
    assert private == {model.__tablename__ for model in db.BROWSER_TABLES}


def test_saved_llm_answers_survive_a_reset(fake_llm):
    browser = new_browser()
    browser.post("/transactions", json=SWIGGY)
    browser.post("/reset", params={"demo": False})
    browser.post("/transactions", json=SWIGGY)
    assert len(fake_llm.calls) == 1


def test_the_committed_seed_has_the_current_tables_and_only_demo_rows():
    assert COMMITTED_SEED.exists()  # checked first, because connecting would create an empty file
    engine = db.sqlite_engine(COMMITTED_SEED)
    inspector = inspect(engine)
    assert set(inspector.get_table_names()) == {model.__tablename__ for model in db.BROWSER_TABLES}
    for model in db.BROWSER_TABLES:
        columns = {column["name"] for column in inspector.get_columns(model.__tablename__)}
        assert columns == set(model.__table__.columns.keys()), model.__name__

    with Session(engine) as session:
        rows = session.exec(select(Transaction)).all()
        assert len(rows) == 325
        assert {row.source for row in rows} == {DEMO_SOURCE}
        assert UNCATEGORIZED not in {row.category for row in rows}
        assert session.exec(select(Budget)).all()
        assert len(session.exec(select(InsightReport)).all()) == 1
