"""Does the LLM read statement layouts and PDFs correctly, and what does each setting change?

Run with:  python -m scripts.eval_import --settings default low off --out answers.json
Re-score:  python -m scripts.eval_import --rescore answers.json   (uses no requests)

One real request per layout, and one per 3 PDF pages, for each setting. All layouts are invented from
general knowledge of bank exports, and the PDF is my own demo layout, not a real PhonePe statement.
"""
import csv
import io
import json
import re
from pathlib import Path

import pandas as pd
import pdfplumber

from app.services.importer import (
    COLUMNS_SCHEMA,
    COLUMNS_SYSTEM,
    PAGES_PER_REQUEST,
    PDF_SCHEMA,
    PDF_SYSTEM,
    read_table,
    rows_from_table,
    sample_rows,
)
from app.services.llm import mask
from scripts.eval_common import parse_args, save_or_load

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
PDF_ROW = re.compile(r"^(\w{3} \d{2}, \d{4})\s+(?:Paid to|Received from) .+?\s+(DEBIT|CREDIT)\s+INR ([\d,.]+)$")


def invented(rows: list[list[str]]) -> bytes:
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    return buffer.getvalue().encode()


# Each: the file, what the layout must be read as, and the rows it must give as (date, amount, type).
LAYOUTS = {
    "debit and credit columns, header on row 5": (
        (SAMPLES / "demo_statement.csv").read_bytes(),
        {"header_row": 5, "date_col": 0, "description_col": 1, "debit_col": 4, "credit_col": 5, "day_first": True},
        None,
    ),
    "description before the money, value date first": (
        invented(
            [
                ["Account Name :", "Demo User"],
                ["Account Number :", "XXXXXXX4321"],
                ["Branch :", "Demo Branch", ""],
                [],
                ["Txn Date", "Value Date", "Description", "Ref No./Cheque No.", "Debit", "Credit", "Balance"],
                ["1 Oct 2026", "1 Oct 2026", "BY TRANSFER-NEFT-ACME TECH SALARY", "REF1", "", "50,000.00", "60,000.00"],
                ["3 Oct 2026", "3 Oct 2026", "TO TRANSFER-UPI-SWIGGY", "REF2", "450.00", "", "59,550.00"],
                ["5 Oct 2026", "5 Oct 2026", "TO TRANSFER-UPI-BIGBASKET", "REF3", "1,200.50", "", "58,349.50"],
                ["7 Oct 2026", "7 Oct 2026", "BY TRANSFER-UPI-REFUND", "REF4", "", "99.00", "58,448.50"],
            ]
        ),
        {"header_row": 4, "date_col": 0, "description_col": 2, "debit_col": 4, "credit_col": 5, "day_first": True},
        [("2026-10-01", 50000.0, "income"), ("2026-10-03", 450.0, "expense"), ("2026-10-05", 1200.5, "expense"), ("2026-10-07", 99.0, "income")],
    ),
    "one amount column with a Cr/Dr column": (
        invented(
            [
                ["Detailed Statement", "", "", "", "", "", "", ""],
                ["Period:", "01/10/2026 - 31/10/2026", "", "", "", "", "", ""],
                ["S No.", "Value Date", "Transaction Date", "Cheque Number", "Transaction Remarks", "Transaction Amount(INR)", "Cr/Dr", "Balance(INR)"],
                ["1", "01/10/2026", "01/10/2026", "-", "NEFT-ACME TECH SALARY", "50000.00", "CR", "60000.00"],
                ["2", "02/10/2026", "02/10/2026", "-", "UPI/ZOMATO/Order", "650.00", "DR", "59350.00"],
                ["3", "13/10/2026", "13/10/2026", "-", "UPI/UBER/Ride", "210.00", "DR", "59140.00"],
            ]
        ),
        # Value date or transaction date are both fair choices, so either column is right.
        {"header_row": 2, "date_col": (1, 2), "description_col": 4, "amount_col": 5, "type_col": 6, "day_first": True},
        [("2026-10-01", 50000.0, "income"), ("2026-10-02", 650.0, "expense"), ("2026-10-13", 210.0, "expense")],
    ),
    "one signed amount column in a UPI app export": (
        invented(
            [
                ["Date", "Time", "Transaction Details", "Other Transaction Details", "Your Account", "Amount", "UPI Ref No.", "Remarks"],
                ["28/10/2026", "21:10", "Paid to Swiggy", "", "Demo Bank - 4321", "-420.00", "REF1", ""],
                ["25/10/2026", "10:02", "Received from Priya Sharma", "", "Demo Bank - 4321", "+800.00", "REF2", "Trip"],
                ["14/10/2026", "18:45", "Paid to Myntra", "", "Demo Bank - 4321", "-1,999.00", "REF3", ""],
            ]
        ),
        {"header_row": 0, "date_col": 0, "description_col": 2, "amount_col": 5, "day_first": True},
        [("2026-10-28", 420.0, "expense"), ("2026-10-25", 800.0, "income"), ("2026-10-14", 1999.0, "expense")],
    ),
    "month-first dates": (
        invented(
            [
                ["Posting Date", "Details", "Withdrawals", "Deposits", "Running Balance"],
                ["10/13/2026", "CARD PURCHASE DEMO STORE", "75.25", "", "924.75"],
                ["10/02/2026", "PAYROLL DEPOSIT", "", "1000.00", "1000.00"],
            ]
        ),
        {"header_row": 0, "date_col": 0, "description_col": 1, "debit_col": 2, "credit_col": 3, "day_first": False},
        [("2026-10-13", 75.25, "expense"), ("2026-10-02", 1000.0, "income")],
    ),
}


def score_layout(name: str, answer: dict) -> list[str]:
    """Check the mapped columns, then read the file with them and compare the rows."""
    data, expected, rows = LAYOUTS[name]
    problems = [
        f"{field}: got {answer.get(field)}, want {value}"
        for field, value in expected.items()
        if answer.get(field) not in (value if isinstance(value, tuple) else (value,))
    ]
    if rows is not None:
        try:
            read, _ = rows_from_table(read_table(".csv", data), answer)
            got = [(str(r["date"]), r["amount"], r["type"]) for r in read]
        except (KeyError, TypeError, ValueError) as exc:
            got = f"could not read rows: {exc!r}"
        if got != rows:
            problems.append(f"rows: got {got}")
    return problems


def pdf_truth() -> tuple[list[str], list[tuple]]:
    with pdfplumber.open(SAMPLES / "demo_upi_statement.pdf") as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    truth = []
    for line in "\n".join(pages).splitlines():
        match = PDF_ROW.match(line.strip())
        if match:
            date = pd.to_datetime(match[1], format="%b %d, %Y").date()
            kind = "expense" if match[2] == "DEBIT" else "income"
            truth.append((str(date), float(match[3].replace(",", "")), kind))
    return pages, truth


def score_pdf(answers: list[dict], truth: list[tuple]) -> dict:
    got = [(item["date"], float(item["amount"]), item["type"]) for answer in answers for item in answer["transactions"]]
    remaining = list(truth)
    invented_rows = []
    for row in got:
        if row in remaining:
            remaining.remove(row)
        else:
            invented_rows.append(row)
    return {"rows": len(got), "right": len(truth) - len(remaining), "missed": remaining, "not_in_pdf": invented_rows}


def main() -> None:
    args = parse_args("Score layout reading and PDF extraction.", ["default", "low", "off"], 1)
    pages, truth = pdf_truth()

    def make_jobs():
        jobs = []
        for setting in args.settings:
            for name, (data, _, _) in LAYOUTS.items():
                sample = sample_rows(read_table(".csv", data))
                user = mask(json.dumps(sample, ensure_ascii=False))
                jobs.append({"setting": setting, "task": name, "system": COLUMNS_SYSTEM, "user": user, "schema": COLUMNS_SCHEMA})
            for start in range(0, len(pages), PAGES_PER_REQUEST):
                text = mask("\n".join(pages[start : start + PAGES_PER_REQUEST]))
                jobs.append({"setting": setting, "task": "pdf", "system": PDF_SYSTEM, "user": text, "schema": PDF_SCHEMA})
        return jobs

    results = save_or_load(args, make_jobs)
    for setting in dict.fromkeys(r["setting"] for r in results):
        print(f"\n=== {setting}")
        chosen = [r for r in results if r["setting"] == setting]
        for result in (r for r in chosen if r["task"] != "pdf"):
            if not result["answer"]:
                print(f"  FAILED {result['task']}: {str(result['error'])[:160]}")
                continue
            problems = score_layout(result["task"], result["answer"])
            print(f"  {result['seconds']:6.1f}s  {'right' if not problems else 'WRONG'}  {result['task']}")
            for problem in problems:
                print(f"      {problem[:200]}")
        parts = [r for r in chosen if r["task"] == "pdf"]
        if parts and all(r["answer"] for r in parts):
            s = score_pdf([r["answer"] for r in parts], truth)
            seconds = [r["seconds"] for r in parts]
            print(f"  {sum(seconds):6.1f}s  PDF {seconds}: {s['right']} of {len(truth)} rows right, {s['rows']} returned")
            print(f"      missed {s['missed'][:5]}")
            print(f"      not in the PDF {s['not_in_pdf'][:5]}")
        elif parts:
            print(f"  FAILED PDF: {[str(r['error'])[:120] for r in parts if r['error']]}")


if __name__ == "__main__":
    main()
