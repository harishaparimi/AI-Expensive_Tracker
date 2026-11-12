import datetime as dt
import io
import re
import threading
import time

import pandas as pd
import pytest

from app.services import importer, llm
from app.services.importer import parse_amount, rows_from_pdf, rows_from_table
from scripts.make_demo_data import write_pdf

CSV = (
    "DEMO BANK\n"
    "Account holder:,Demo User\n"
    "Date,Narration,Debit,Credit,Balance\n"
    "03/05/26,UPI-SWIGGY-swiggy@ybl-612345678901-PAYMENT,412.00,,10000.00\n"
    "04/05/26,UPI-SOME SHOP-shop@ybl-612345678902-PAYMENT,\"1,250.50\",,8749.50\n"
    "05/05/26,SALARY,,\"65,000.00\",73749.50\n"
    "Total,,,,\n"
    ",,,,\n"
)


def upload(client, name, data):
    return client.post("/import", files={"file": (name, data)})


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1,00,000.50", 100000.50),
        ("Rs. 350", 350.0),
        ("INR 1,200.00", 1200.0),
        ("-120", -120.0),
        ("(45.50)", -45.5),
        ("1,200.00 Cr", 1200.0),
        ("", None),
        ("abc", None),
        (None, None),
    ],
)
def test_parse_amount(value, expected):
    assert parse_amount(value) == expected


def test_csv_import_reads_rows_and_skips_header_and_footer(client, fake_llm):
    result = upload(client, "statement.csv", CSV.encode()).json()
    assert result == {
        "imported": 3,
        "duplicates": 0,
        "skipped": 1,
        "uncategorized": 0,
        "warning": None,
        "replaced_demo_data": False,
    }

    rows = {t["description"]: t for t in client.get("/transactions").json()}
    swiggy = rows["UPI-SWIGGY-swiggy@ybl-612345678901-PAYMENT"]
    assert swiggy["date"] == "2026-05-03"  # day first
    assert (swiggy["amount"], swiggy["type"]) == (412.0, "expense")
    assert (swiggy["merchant"], swiggy["category"], swiggy["source"]) == ("Swiggy", "Food & Dining", "csv")
    assert (rows["SALARY"]["amount"], rows["SALARY"]["type"]) == (65000.0, "income")


def test_personal_details_are_masked_before_sending(client, fake_llm):
    upload(client, "statement.csv", CSV.encode())
    sent = "\n".join(fake_llm.calls)
    assert "swiggy@ybl" not in sent and "612345678901" not in sent
    assert "Demo User" not in sent  # the short account rows above the table are left out


def test_importing_the_same_file_twice_adds_nothing(client, fake_llm):
    upload(client, "statement.csv", CSV.encode())
    calls_after_first = len(fake_llm.calls)
    result = upload(client, "statement.csv", CSV.encode()).json()
    assert (result["imported"], result["duplicates"]) == (0, 3)
    assert len(client.get("/transactions").json()) == 3
    assert len(fake_llm.calls) == calls_after_first  # no new LLM requests


def test_identical_rows_in_one_file_are_both_kept(client, fake_llm):
    row = "03/05/26,UPI-SWIGGY-TEA,20.00,,100.00\n"
    data = ("Date,Narration,Debit,Credit,Balance\n" + row + row).encode()
    fake_llm.columns = dict(fake_llm.columns, header_row=0)
    assert upload(client, "tea.csv", data).json()["imported"] == 2
    assert upload(client, "tea.csv", data).json()["duplicates"] == 2


def test_rows_differing_only_by_reference_number_use_one_llm_entry(client, fake_llm):
    data = (
        "Date,Narration,Debit,Credit,Balance\n"
        "03/05/26,UPI-SWIGGY-111111111111,100.00,,1\n"
        "04/05/26,UPI-SWIGGY-222222222222,200.00,,1\n"
    ).encode()
    fake_llm.columns = dict(fake_llm.columns, header_row=0)
    upload(client, "swiggy.csv", data)
    assert fake_llm.calls[-1].count('"id"') == 1
    assert all(t["merchant"] == "Swiggy" for t in client.get("/transactions").json())


@pytest.mark.parametrize("date_cell", ["03/05/2026", dt.datetime(2026, 5, 3)], ids=["text_date", "excel_date"])
def test_xlsx_import_keeps_day_and_month_right(client, fake_llm, date_cell):
    buffer = io.BytesIO()
    pd.DataFrame(
        [["Date", "Narration", "Debit", "Credit", "Balance"], [date_cell, "UPI-SWIGGY", "412.00", None, "100.00"]]
    ).to_excel(buffer, header=False, index=False)
    fake_llm.columns = dict(fake_llm.columns, header_row=0)
    assert upload(client, "statement.xlsx", buffer.getvalue()).json()["imported"] == 1
    assert client.get("/transactions").json()[0]["date"] == "2026-05-03"


@pytest.mark.parametrize(
    "cells, expected",
    [
        (["2026-05-03", "x", "-500", None], ("2026-05-03", 500.0, "expense")),
        (["2026-05-03", "x", "500", None], ("2026-05-03", 500.0, "income")),
        (["03-May-2026", "x", "500", "DR"], ("2026-05-03", 500.0, "expense")),
        (["03-May-2026", "x", "500", "Credit"], ("2026-05-03", 500.0, "income")),
        (["03/05/2026", "x", "1,200.00 Cr", None], ("2026-05-03", 1200.0, "income")),
        (["03/05/2026", "x", "1,200.00 Dr", None], ("2026-05-03", 1200.0, "expense")),
    ],
)
def test_single_amount_column_layouts(cells, expected):
    table = pd.DataFrame([["Date", "Details", "Amount", "Type"], cells])
    columns = {
        "header_row": 0,
        "date_col": 0,
        "description_col": 1,
        "debit_col": None,
        "credit_col": None,
        "amount_col": 2,
        "type_col": 3,
        "day_first": True,
    }
    rows, skipped = rows_from_table(table, columns)
    assert skipped == 0
    assert (str(rows[0]["date"]), rows[0]["amount"], rows[0]["type"]) == expected


def make_pdf(tmp_path, lines, lines_per_page=48):
    path = tmp_path / "statement.pdf"
    write_pdf(path, lines, lines_per_page)
    return path.read_bytes()


def one_page_each(tmp_path, count, first_amount=100):
    """A PDF with one transaction line per page, each with its own amount."""
    lines = [f"Oct 03, 2026 Paid INR {first_amount + i}.00" for i in range(count)]
    return make_pdf(tmp_path, lines, lines_per_page=1)


def answer_from_text(user):
    """Answer a PDF request the way a good model would: one row per amount in the text."""
    amounts = re.findall(r"INR (\d+)\.00", user)
    rows = [{"date": "2026-10-03", "description": f"Row {a}", "amount": float(a), "type": "expense"} for a in amounts]
    return {"transactions": rows}, "fake-model"


def test_pdf_requests_run_at_the_same_time(fake_llm, monkeypatch, tmp_path):
    """The barrier only opens with 3 requests in flight, so sending them in turn would fail."""
    barrier = threading.Barrier(3, timeout=10)

    def fake(system, user, schema, models):
        barrier.wait()
        return answer_from_text(user)

    monkeypatch.setattr(llm, "call_openrouter", fake)
    rows, skipped = rows_from_pdf(one_page_each(tmp_path, 9))
    assert (len(rows), skipped) == (9, 0)


def test_pdf_rows_keep_page_order_when_answers_arrive_out_of_order(fake_llm, monkeypatch, tmp_path):
    def fake(system, user, schema, models):
        if "INR 100.00" in user:  # the first pages answer last
            time.sleep(0.3)
        return answer_from_text(user)

    monkeypatch.setattr(llm, "call_openrouter", fake)
    rows, _ = rows_from_pdf(one_page_each(tmp_path, 9))
    assert [r["amount"] for r in rows] == [100.0 + i for i in range(9)]


def test_a_pdf_amount_must_be_in_the_pages_sent_with_it(fake_llm, monkeypatch, tmp_path):
    """An amount from another request's pages is not evidence, even though it is in the same PDF."""

    def fake(system, user, schema, models):
        answer, model = answer_from_text(user)
        if "INR 100.00" in user:
            answer["transactions"].append(
                {"date": "2026-10-03", "description": "From later pages", "amount": 108.0, "type": "expense"}
            )
        return answer, model

    monkeypatch.setattr(llm, "call_openrouter", fake)
    rows, skipped = rows_from_pdf(one_page_each(tmp_path, 9))
    assert (len(rows), skipped) == (9, 1)


def test_a_failed_pdf_request_saves_nothing_and_a_retry_resends_only_it(client, fake_llm, monkeypatch, tmp_path):
    sent = []
    failing = {"on": True}

    def fake(system, user, schema, models):
        if "transactions" not in schema["properties"]:
            return fake_llm(system, user, schema, models)
        sent.append(user)
        if failing["on"] and "INR 103.00" in user:
            raise llm.LLMError("Rate limit reached.")
        return answer_from_text(user)

    monkeypatch.setattr(llm, "call_openrouter", fake)
    data = one_page_each(tmp_path, 9)

    assert upload(client, "statement.pdf", data).status_code == 502
    assert client.get("/transactions").json() == []
    assert len(sent) == 3

    sent.clear()
    failing["on"] = False
    assert upload(client, "statement.pdf", data).json()["imported"] == 9
    assert len(sent) == 1 and "INR 103.00" in sent[0]


def test_a_pdf_over_the_page_limit_is_rejected_before_any_request(client, fake_llm, monkeypatch, tmp_path):
    monkeypatch.setattr(importer, "MAX_PDF_PAGES", 3)
    response = upload(client, "statement.pdf", one_page_each(tmp_path, 4))
    assert response.status_code == 400
    assert "4 pages" in response.json()["detail"] and "3" in response.json()["detail"]
    assert fake_llm.calls == []


def test_a_pdf_at_the_page_limit_is_accepted(fake_llm, monkeypatch, tmp_path):
    monkeypatch.setattr(importer, "MAX_PDF_PAGES", 3)
    monkeypatch.setattr(llm, "call_openrouter", lambda system, user, schema, models: answer_from_text(user))
    rows, _ = rows_from_pdf(one_page_each(tmp_path, 3))
    assert len(rows) == 3


def test_blank_pages_use_no_request(fake_llm, monkeypatch, tmp_path):
    sent = []

    def fake(system, user, schema, models):
        sent.append(user)
        return answer_from_text(user)

    monkeypatch.setattr(llm, "call_openrouter", fake)
    data = make_pdf(tmp_path, ["", "", "", "Oct 03, 2026 Paid INR 100.00"], lines_per_page=1)
    rows, _ = rows_from_pdf(data)
    assert len(rows) == 1 and len(sent) == 1


def test_pdf_import_drops_amounts_not_in_the_source(client, fake_llm, tmp_path):
    data = make_pdf(tmp_path, ["Oct 03, 2026 Paid to SWIGGY DEBIT INR 1,412.50", "Oct 04, 2026 Received CREDIT INR 500"])
    fake_llm.pdf_transactions = [
        {"date": "2026-10-03", "description": "Paid to SWIGGY", "amount": 1412.5, "type": "expense"},
        {"date": "2026-10-04", "description": "Received", "amount": 500, "type": "income"},
        {"date": "2026-10-05", "description": "Invented row", "amount": 7777.77, "type": "expense"},
        {"date": "not a date", "description": "Bad date", "amount": 500, "type": "expense"},
    ]
    result = upload(client, "statement.pdf", data).json()
    assert (result["imported"], result["skipped"]) == (2, 2)
    assert {t["description"] for t in client.get("/transactions").json()} == {"Paid to SWIGGY", "Received"}


def test_same_payment_in_two_statements_counts_once(client, fake_llm, tmp_path):
    upload(client, "statement.csv", CSV.encode())  # has 412.00 on 2026-05-03
    fake_llm.pdf_transactions = [
        {"date": "2026-05-03", "description": "Paid to SWIGGY", "amount": 412.0, "type": "expense"},
        {"date": "2026-05-09", "description": "Paid to SWIGGY", "amount": 412.0, "type": "expense"},
    ]
    data = make_pdf(tmp_path, ["May 03, 2026 Paid to SWIGGY DEBIT INR 412.00", "May 09, 2026 Paid to SWIGGY DEBIT INR 412.00"])
    result = upload(client, "upi_app.pdf", data).json()
    assert (result["imported"], result["duplicates"]) == (1, 1)  # only the other date is new
    assert len(client.get("/transactions").json()) == 4


def test_cross_file_match_is_used_only_once(client, fake_llm, tmp_path):
    upload(client, "statement.csv", CSV.encode())
    fake_llm.pdf_transactions = [
        {"date": "2026-05-03", "description": "First", "amount": 412.0, "type": "expense"},
        {"date": "2026-05-03", "description": "Second", "amount": 412.0, "type": "expense"},
    ]
    data = make_pdf(tmp_path, ["May 03, 2026 INR 412.00"])
    result = upload(client, "upi_app.pdf", data).json()
    assert (result["imported"], result["duplicates"]) == (1, 1)  # one bank row matches one app row


def test_manual_entries_are_not_treated_as_imported_duplicates(client, fake_llm):
    client.post(
        "/transactions",
        json={"date": "2026-05-03", "description": "Lunch", "amount": 412.0, "type": "expense", "category": "Food & Dining"},
    )
    assert upload(client, "statement.csv", CSV.encode()).json()["imported"] == 3


def test_unsupported_file_type_is_rejected(client, fake_llm):
    response = upload(client, "statement.txt", b"hello")
    assert response.status_code == 400
    assert fake_llm.calls == []


@pytest.mark.parametrize("name", ["statement.pdf", "statement.xlsx"])
def test_broken_files_get_a_clear_error(client, fake_llm, name):
    response = upload(client, name, b"this is not a real file")
    assert response.status_code == 400
    assert fake_llm.calls == []


def test_layout_failure_returns_502_and_saves_nothing(client, fake_llm):
    fake_llm.fail = True
    response = upload(client, "statement.csv", CSV.encode())
    assert response.status_code == 502
    assert client.get("/transactions").json() == []


def test_categorizing_failure_keeps_rows_and_recategorize_fills_them(client, fake_llm):
    fake_llm.fail_categorizing = True
    result = upload(client, "statement.csv", CSV.encode()).json()
    assert (result["imported"], result["uncategorized"]) == (3, 3)
    assert "Rate limit" in result["warning"]

    fake_llm.fail_categorizing = False
    again = client.post("/transactions/recategorize").json()
    assert again == {"tried": 3, "still_uncategorized": 0, "warning": None}
    assert all(t["category"] != "Uncategorized" for t in client.get("/transactions").json())


@pytest.mark.parametrize("header_row", [1, 2, 3])
def test_a_header_row_one_out_loses_no_transactions(header_row):
    """Real answers are sometimes a row out, and the first transaction must not be dropped."""
    table = pd.DataFrame(
        [
            ["Statement", None, None],
            ["Period", "Oct", None],
            ["Date", "Narration", "Amount"],
            ["01/10/2026", "SWIGGY", "-100.00"],
            ["02/10/2026", "SALARY", "500.00"],
        ]
    )
    columns = {
        "header_row": header_row,
        "date_col": 0,
        "description_col": 1,
        "debit_col": None,
        "credit_col": None,
        "amount_col": 2,
        "type_col": None,
        "day_first": True,
    }
    rows, _ = rows_from_table(table, columns)
    assert [r["description"] for r in rows] == ["SWIGGY", "SALARY"]
