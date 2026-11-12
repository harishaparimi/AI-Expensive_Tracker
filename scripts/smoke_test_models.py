"""Compares the free models in LLM_MODELS on synthetic data.

Run with: python -m scripts.smoke_test_models
Makes 3 real requests per model and ignores saved answers, so it counts toward the daily cap.
"""
import json
import sys
import time
from pathlib import Path

from app import config
from app.services.categorizer import SCHEMA as CATEGORY_SCHEMA
from app.services.categorizer import SYSTEM as CATEGORY_SYSTEM
from app.services.categorizer import build_prompt
from app.services.importer import COLUMNS_SCHEMA, COLUMNS_SYSTEM, read_table, sample_rows
from app.services.llm import LLMError, call_openrouter, mask

DEMO_CSV = Path(__file__).resolve().parent.parent / "samples" / "demo_statement.csv"
EXPECTED_COLUMNS = {"header_row": 5, "date_col": 0, "description_col": 1, "debit_col": 4, "credit_col": 5, "day_first": True}

# narration, type, a word the merchant should contain, acceptable categories
CASES = [
    ("UPI-SWIGGY-swiggy@ybl-YESB0000123-612345678901-PAYMENT", "expense", "swiggy", ["Food & Dining"]),
    ("UPI-BIGBASKET-bigbasket@axisbank-AXIS0000456-612345678902-PAYMENT", "expense", "bigbasket", ["Groceries"]),
    ("UPI-UBER INDIA-uber@axisbank-AXIS0000789-612345678903-PAYMENT", "expense", "uber", ["Transport"]),
    ("NEFT DR-SBIN0001234-MRS DEMO LANDLORD-RENT-676846326096", "expense", "landlord", ["Rent & Housing"]),
    ("UPI-NETFLIX-netflix@icici-ICIC0000011-612345678904-MONTHLY", "expense", "netflix", ["Subscriptions"]),
    ("ACH D- GROWW MUTUAL FUND SIP-616048021673", "expense", "groww", ["Investments"]),
    ("ACH C- NIMBUS SOFTECH PVT LTD SALARY MAY-615631219101", "income", "nimbus", ["Income"]),
    ("UPI-RAHUL DEMO-rahul.demo@oksbi-SBIN0000066-612345678905-SENT", "expense", "rahul", ["Transfers"]),
    ("BILLPAY BESCOM ELECTRICITY-612345678906", "expense", "bescom", ["Bills & Utilities"]),
    ("UPI-APOLLO PHARMACY-apollo@hdfcbank-HDFC0000111-612345678907-PAYMENT", "expense", "apollo", ["Health"]),
    ("UPI-MYNTRA-myntra@icici-ICIC0000222-612345678908-PAYMENT", "expense", "myntra", ["Shopping"]),
    ("UPI-CULT FIT GYM-cultfit@hdfcbank-HDFC0000044-612345678909-MEMBERSHIP", "expense", "cult", ["Health", "Subscriptions"]),
    ("UPI-IRCTC-irctc@sbi-SBIN0000333-612345678910-TICKET", "expense", "irctc", ["Travel", "Transport"]),
    ("ATW-416021XXXXXX1234-S1AWBL05-BENGALURU-612345678911", "expense", "", ["Other", "Transfers"]),
]

INSIGHT_SYSTEM = (
    "You are a personal finance coach in India. Using only the spending summary, give three "
    "specific ways to cut spending. Each must name a merchant from the summary and its real amount."
)
INSIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "merchant": {"type": "string"},
                    "spent": {"type": "number"},
                    "advice": {"type": "string"},
                },
                "required": ["merchant", "spent", "advice"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["suggestions"],
    "additionalProperties": False,
}
INSIGHT_SUMMARY = {
    "month": "October 2026",
    "income": 65500,
    "spent": 67183,
    "by_merchant": [
        {"merchant": "Swiggy", "orders": 14, "total": 6020, "last_month_total": 4410},
        {"merchant": "Zomato", "orders": 10, "total": 3950, "last_month_total": 2980},
        {"merchant": "Myntra", "orders": 3, "total": 7200, "last_month_total": 1450},
        {"merchant": "Third Wave Coffee", "orders": 8, "total": 3180, "last_month_total": 2340},
        {"merchant": "Cult Fit Gym", "orders": 1, "total": 1499, "last_month_total": 1499},
    ],
}


def timed_call(system, user, schema, model):
    started = time.perf_counter()
    answer, served_by = call_openrouter(system, user, schema, [model])
    return answer, served_by, time.perf_counter() - started


def test_categorizing(model):
    items = [{"id": i, "text": mask(text), "type": kind} for i, (text, kind, _, _) in enumerate(CASES)]
    answer, _, seconds = timed_call(CATEGORY_SYSTEM, build_prompt(items, []), CATEGORY_SCHEMA, model)
    answers = {item["id"]: item for item in answer["items"]}

    right = 0
    for i, (text, _, merchant_word, categories) in enumerate(CASES):
        item = answers.get(i, {})
        is_right = item.get("category") in categories and merchant_word in item.get("merchant", "").lower()
        right += is_right
        if not is_right:
            print(f"    wrong: {text[:45]}... got {item.get('merchant')!r} / {item.get('category')!r}")
    print(f"  categorizing: {right}/{len(CASES)} right in {seconds:.1f}s")


def test_columns(model):
    sample = sample_rows(read_table(".csv", DEMO_CSV.read_bytes()))
    answer, _, seconds = timed_call(COLUMNS_SYSTEM, mask(json.dumps(sample)), COLUMNS_SCHEMA, model)
    wrong = {field: answer.get(field) for field, value in EXPECTED_COLUMNS.items() if answer.get(field) != value}
    print(f"  column layout: {'right' if not wrong else f'WRONG {wrong}'} in {seconds:.1f}s")


def test_insights(model):
    answer, _, seconds = timed_call(INSIGHT_SYSTEM, json.dumps(INSIGHT_SUMMARY), INSIGHT_SCHEMA, model)
    print(f"  insights in {seconds:.1f}s:")
    for suggestion in answer["suggestions"]:
        print(f"    {suggestion['merchant']} (Rs {suggestion['spent']}): {suggestion['advice']}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console cannot otherwise print the rupee sign
    for model in config.LLM_MODELS:
        print(f"\n{model}")
        for test in (test_categorizing, test_columns, test_insights):
            try:
                test(model)
            except (LLMError, KeyError, TypeError) as exc:
                print(f"  {test.__name__} FAILED: {exc!r}")


if __name__ == "__main__":
    main()
