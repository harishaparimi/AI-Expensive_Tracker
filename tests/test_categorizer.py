"""The LLM is asked about each description once, whichever file or request it first appeared in."""
import csv
import io
import json
from pathlib import Path

from sqlmodel import Session, select

from app import db
from app.models import DescriptionAnswer
from app.services import categorizer, llm

DEMO_CSV = Path(__file__).resolve().parent.parent / "samples" / "demo_statement.csv"
DEMO_COLUMNS = {
    "header_row": 0,
    "date_col": 0,
    "description_col": 1,
    "debit_col": 4,
    "credit_col": 5,
    "amount_col": None,
    "type_col": None,
    "day_first": True,
}


def add(client, description, kind="expense"):
    body = {"date": "2026-10-03", "description": description, "amount": 250, "type": kind}
    return client.post("/transactions", json=body).json()


def sent_items(fake_llm):
    """Every description sent for categorizing. A layout request is a plain list of rows."""
    prompts = [json.loads(call) for call in fake_llm.calls]
    return [item["text"] for prompt in prompts if isinstance(prompt, dict) for item in prompt["items"]]


def reply_with(monkeypatch, *answers):
    """Make the LLM give these answers in order. Each entry is the list of items it returns."""
    calls = []
    queue = list(answers)

    def fake(system, user, schema, models):
        calls.append(user)
        return {"items": queue.pop(0)}, "fake-model"

    monkeypatch.setattr(llm, "call_openrouter", fake)
    return calls


def test_a_repeated_description_uses_no_request(client, fake_llm):
    first = add(client, "SWIGGY order")
    second = add(client, "SWIGGY order")
    assert len(fake_llm.calls) == 1
    assert (second["merchant"], second["category"], second["category_source"]) == (
        first["merchant"],
        first["category"],
        "llm",
    )


def test_a_reference_number_does_not_make_a_description_new(client, fake_llm):
    add(client, "UPI-SWIGGY-111111111111")
    add(client, "UPI-SWIGGY-222222222222")
    assert len(fake_llm.calls) == 1


def test_only_unseen_descriptions_are_sent(client, fake_llm):
    add(client, "SWIGGY order")
    fake_llm.calls.clear()
    fake_llm.columns = dict(fake_llm.columns, header_row=0)
    data = (
        "Date,Narration,Debit,Credit,Balance\n"
        "03/05/26,SWIGGY order,100.00,,1\n"
        "04/05/26,NEW SHOP,200.00,,1\n"
    ).encode()
    assert client.post("/import", files={"file": ("a.csv", data)}).json()["imported"] == 2
    assert sent_items(fake_llm) == ["NEW SHOP"]


def test_october_after_earlier_months_sends_only_what_is_new(client, fake_llm):
    """Measured on the demo statement: this used to resend all 23 descriptions when 22 were known."""
    rows = list(csv.reader(DEMO_CSV.read_text(encoding="utf-8-sig").splitlines()))
    header_at = next(i for i, row in enumerate(rows) if row and row[0].strip().lower() == "date")
    header, body = rows[header_at], [row for row in rows[header_at + 1 :] if row and row[0][:2].isdigit()]

    def statement(wanted):
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(wanted)
        return out.getvalue().encode()

    # Dates are day first: 03/10/2026 is in October.
    october = [row for row in body if row[0].split("/")[1] == "10"]
    earlier = [row for row in body if row[0].split("/")[1] != "10"]
    assert october and earlier

    fake_llm.columns = DEMO_COLUMNS
    client.post("/import", files={"file": ("earlier.csv", statement(earlier))})
    fake_llm.calls.clear()
    result = client.post("/import", files={"file": ("october.csv", statement(october))}).json()

    assert result["imported"] == len(october)
    assert result["uncategorized"] == 0
    assert len(sent_items(fake_llm)) == 1


def test_same_text_with_another_type_is_asked_separately(client, fake_llm):
    add(client, "SWIGGY order")
    add(client, "SWIGGY order", kind="income")  # a refund can deserve a different answer
    assert len(fake_llm.calls) == 2


def test_a_users_correction_beats_the_remembered_answer(client, fake_llm):
    first = add(client, "SWIGGY order")
    client.patch(f"/transactions/{first['id']}", json={"category": "Groceries"})
    fake_llm.calls.clear()

    later = add(client, "SWIGGY order")
    assert (later["category"], later["category_source"]) == ("Groceries", "user")
    assert fake_llm.calls == []


def test_deleting_a_row_keeps_the_answer(client, fake_llm):
    first = add(client, "SWIGGY order")
    client.delete(f"/transactions/{first['id']}")
    add(client, "SWIGGY order")
    assert len(fake_llm.calls) == 1


def test_a_retry_asks_again_after_the_model_left_a_row_out(client, monkeypatch):
    """An answer with no row for an id must not be replayed, or the row can never be categorized."""
    answered = [{"id": 0, "merchant": "Swiggy", "category": "Food & Dining"}]
    calls = reply_with(monkeypatch, [], answered)

    created = add(client, "SWIGGY order")
    assert created["category"] == "Uncategorized"

    again = client.post("/transactions/recategorize").json()
    assert again == {"tried": 1, "still_uncategorized": 0, "warning": None}
    assert len(calls) == 2


def test_an_answer_with_a_bad_category_is_not_remembered(client, monkeypatch):
    bad = [{"id": 0, "merchant": "Swiggy", "category": "Food"}]
    good = [{"id": 0, "merchant": "Swiggy", "category": "Food & Dining"}]
    calls = reply_with(monkeypatch, bad, good)

    assert add(client, "SWIGGY order")["category"] == "Other"
    with Session(db.shared_engine) as session:
        assert session.exec(select(DescriptionAnswer)).all() == []

    add(client, "SWIGGY order")
    assert len(calls) == 2


def test_an_answer_with_a_blank_merchant_is_not_remembered(client, monkeypatch):
    calls = reply_with(
        monkeypatch,
        [{"id": 0, "merchant": "  ", "category": "Shopping"}],
        [{"id": 0, "merchant": "Some Shop", "category": "Shopping"}],
    )
    assert add(client, "SHOP order")["merchant"] == "Unknown"
    assert add(client, "SHOP order")["merchant"] == "Some Shop"
    assert len(calls) == 2


def test_unknown_is_a_real_answer_and_is_remembered(client, monkeypatch):
    calls = reply_with(monkeypatch, [{"id": 0, "merchant": "Unknown", "category": "Other"}])
    assert add(client, "UPI PAYMENT")["merchant"] == "Unknown"
    assert add(client, "UPI PAYMENT")["category"] == "Other"
    assert len(calls) == 1


def test_a_failed_batch_keeps_earlier_answers_and_a_retry_sends_only_the_rest(client, fake_llm, monkeypatch):
    monkeypatch.setattr(categorizer, "BATCH_SIZE", 2)
    fake_llm.columns = dict(fake_llm.columns, header_row=0)
    data = (
        "Date,Narration,Debit,Credit,Balance\n"
        "03/05/26,SHOP ONE,100.00,,1\n"
        "04/05/26,SHOP TWO,200.00,,1\n"
        "05/05/26,SHOP THREE,300.00,,1\n"
    ).encode()

    answer_categorizing = fake_llm.__call__
    categorizing_calls = []

    def second_batch_fails(system, user, schema, models):
        if "items" in schema["properties"]:
            categorizing_calls.append(user)
            if len(categorizing_calls) == 2:
                raise llm.LLMError("Rate limit reached.")
        return answer_categorizing(system, user, schema, models)

    monkeypatch.setattr(llm, "call_openrouter", second_batch_fails)
    result = client.post("/import", files={"file": ("a.csv", data)}).json()
    assert (result["imported"], result["uncategorized"]) == (3, 1)
    assert "Rate limit" in result["warning"]

    categorizing_calls.clear()
    again = client.post("/transactions/recategorize").json()
    assert again == {"tried": 1, "still_uncategorized": 0, "warning": None}
    assert len(categorizing_calls) == 1
    assert [item["text"] for item in json.loads(categorizing_calls[0])["items"]] == ["SHOP THREE"]


def test_known_names_are_sent_so_spellings_stay_the_same(client, fake_llm):
    add(client, "SWIGGY order")
    add(client, "NEW SHOP")
    assert json.loads(fake_llm.calls[-1])["known_merchants"] == ["Swiggy"]


def test_a_merchants_first_spelling_is_kept(client, monkeypatch):
    reply_with(
        monkeypatch,
        [{"id": 0, "merchant": "SWIGGY", "category": "Food & Dining"}],
        [{"id": 0, "merchant": "Swiggy", "category": "Food & Dining"}],
    )
    add(client, "SWIGGY order")
    assert add(client, "Swiggy Ltd order")["merchant"] == "SWIGGY"


def test_the_llms_category_stands_under_a_known_merchant(client, monkeypatch):
    """Measured: the model files Instamart under Swiggy but still says Groceries, and that is right."""
    reply_with(
        monkeypatch,
        [{"id": 0, "merchant": "Swiggy", "category": "Food & Dining"}],
        [{"id": 0, "merchant": "Swiggy", "category": "Groceries"}],
    )
    add(client, "SWIGGY order")
    instamart = add(client, "SWIGGY INSTAMART")
    assert (instamart["merchant"], instamart["category"], instamart["category_source"]) == ("Swiggy", "Groceries", "llm")


def test_peoples_names_are_not_sent_again(client, monkeypatch):
    calls = reply_with(
        monkeypatch,
        [{"id": 0, "merchant": "Rahul Verma", "category": "Transfers"}],
        [{"id": 0, "merchant": "Anita Rao", "category": "Rent & Housing"}],
        [{"id": 0, "merchant": "Swiggy", "category": "Food & Dining"}],
        [{"id": 0, "merchant": "Some Shop", "category": "Shopping"}],
    )
    for description in ("UPI-RAHUL", "NEFT-ANITA-RENT", "SWIGGY order", "SHOP"):
        add(client, description)
    assert json.loads(calls[-1])["known_merchants"] == ["Swiggy"]


def test_a_long_history_sends_only_the_newest_names(client, monkeypatch):
    monkeypatch.setattr(categorizer, "MAX_KNOWN_NAMES", 2)
    names = ("One", "Two", "Three", "Four")
    calls = reply_with(monkeypatch, *[[{"id": 0, "merchant": name, "category": "Shopping"}] for name in names])
    for description in ("first", "second", "third", "fourth"):
        add(client, description)
    assert json.loads(calls[-1])["known_merchants"] == ["Two", "Three"]


def test_names_from_an_earlier_batch_go_with_the_next_one(client, fake_llm, monkeypatch):
    monkeypatch.setattr(categorizer, "BATCH_SIZE", 1)
    fake_llm.columns = dict(fake_llm.columns, header_row=0)
    data = (
        "Date,Narration,Debit,Credit,Balance\n"
        "03/05/26,SWIGGY order,100.00,,1\n"
        "04/05/26,NEW SHOP,200.00,,1\n"
    ).encode()
    client.post("/import", files={"file": ("a.csv", data)})
    assert json.loads(fake_llm.calls[-1])["known_merchants"] == ["Swiggy"]
