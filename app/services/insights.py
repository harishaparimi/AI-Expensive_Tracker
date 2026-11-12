"""AI insights: pandas works out the facts, the LLM makes the judgement calls and writes the
advice, and Python checks the answer and does all the money arithmetic."""
import calendar
import datetime as dt
import json
import math
import re
from statistics import mean

import pandas as pd
from sqlmodel import Session, select

from app.models import Budget, InsightReport
from app.services import analytics, market
from app.services.llm import ask_llm

MONTHS_TO_SEND = 6
AVERAGE_OVER_MONTHS = 3
MERCHANTS_TO_SEND = 25
PAYMENTS_TO_SEND = 8
GROWING_TO_SEND = 5
MAX_CUTS = 6
DAYS_NEAR_MONTH_END = 3
EMERGENCY_FUND_MONTHS = 6
TERM_COVER_YEARS_OF_INCOME = (10, 15)

PRODUCTS = ["Emergency fund", "Term insurance", "Equity mutual fund", "Gold"]

# A line break, real or written out as backslash-n, because free models produce both.
LINE_BREAK = r"(?:\r?\n|\\n)"
# Where a pasted copy of an answer starts: a JSON object, or a blank line before plain text.
PASTE_START = re.compile(rf'\{{\s*"|{LINE_BREAK}\s*{LINE_BREAK}')
# A name like left_over, which a sentence can only have copied from the instructions or the facts.
FIELD_NAME = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

DISCLAIMER = (
    "For education only, and not advice from a SEBI-registered adviser. Past returns do not predict "
    "future returns. Term cover is a rule of thumb, not an insurance quote."
)

SYSTEM = f"""You are a personal finance coach for someone in India. The user message is JSON facts worked out from their own bank and UPI history. All amounts are in rupees; write them with Indian digit grouping, like Rs 4,500 or Rs 3,23,572.
Use only these facts. Copy figures exactly, and never invent a merchant, an amount or a return. Write plain, direct English that calls the user 'you'. Names with an underscore, such as left_over or data_through, are for you only, so never write one in the text.

Return these fields:

summary: two or three sentences in one paragraph on how the money has moved over the months, with figures from monthly_totals. Give the figures without explaining what caused them. left_over is all money in minus all money out; write it as 'money left over'.

spending_cuts: at most {MAX_CUTS} cuts, each one a real saving. Fewer is fine, so never add a cut just to fill the list.
- merchant: exactly one name from the merchants list, spelled as it is there.
- finding: start with the merchant's name, then what its actual monthly amounts show, for example '<merchant> rose from Rs 1,200 in May to Rs 3,400 in October, with 14 payments in October'. These amounts are the merchant's own, so never present them as its category's. You may add that the category is over its budget.
- percent_to_cut: how much of the merchant's average_last_3_months to cut, from 1 to 100, where 100 means stopping. Aim for a real change, not a token one. The app works out the rupee target from this and shows it.
- action: one specific step that makes that cut, such as how often to order, not 'spend less'. Do not state a rupee target.
Look across the whole list: one-off spikes in largest_payments_latest_month, spending that grows month after month, small habits with many payments, and recurring memberships. Investing is good, so never suggest cutting it.

investment_advice: exactly one item per product, in this order: Emergency fund, Term insurance, Equity mutual fund, Gold. The app works out the money available per month and shows the rupee amounts, so you only choose the split.
- percent_of_available: this product's share of that money, from 0 to 100. The four together must total 100.
- advice: two or three sentences. Do not state a rupee amount per month, and do not mention this product's share.
The invested figure in monthly_totals is what the user already invests each month, and the money left over is already after it, so this plan comes on top. Tell the user this once, in your own words, in the Emergency fund advice, rather than telling them to start from nothing.
Emergency fund comes first and gets most of the money until the target is reached. State the target in rupees, from emergency_fund_target, and say it is {EMERGENCY_FUND_MONTHS} months of average spending. The history does not show existing savings, so say the target is reduced by whatever the user already holds.
Term insurance: give the cover range from term_cover_low to term_cover_high and suggest comparing quotes. No premium data is available, so do not state a premium, and set percent_of_available to 0.
Equity mutual fund and Gold: quote the 1-year and 3-year returns from market and say past returns do not predict future returns. Name the fund as an example only. If a product is missing from market, say that live figures are unavailable."""

# Added only when they apply, because Python decides. As "if this, say that" lines in the prompt,
# the model mentioned a complete month in 5 of 6 runs, and once claimed a deficit that did not exist.
PARTIAL_MONTH_LINE = "The statement stops partway through the latest month, so say in the summary that the latest month is only partly covered."
DEFICIT_LINE = "The money left over averaged below zero over the last 3 months, so say in the Emergency fund advice that fixing spending comes before investing."
MOVED_MONEY_LINE = "In {months}, money left over went negative even though spending stayed below income, because of money invested or transferred. Say that in the summary, and never say spending went above income."
GROWING_LINE = "fastest_growing lists the merchants whose average monthly spending rose most. Every one of them gets a cut, before any other merchant."
RECURRING_LINE = "recurring_payments lists charges of the same amount every month. Give {merchant} a cut. Its finding says the charge is the same every month and states its yearly_cost as the cost per year. Its percent_to_cut is 100, and its action is to cancel it unless it gets regular use, because the history cannot show how much it is used."

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "spending_cuts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "merchant": {"type": "string"},
                    "finding": {"type": "string"},
                    "percent_to_cut": {"type": "number"},
                    "action": {"type": "string"},
                },
                "required": ["merchant", "finding", "percent_to_cut", "action"],
                "additionalProperties": False,
            },
        },
        "investment_advice": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "enum": PRODUCTS},
                    "percent_of_available": {"type": "number"},
                    "advice": {"type": "string"},
                },
                "required": ["product", "percent_of_available", "advice"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "spending_cuts", "investment_advice"],
    "additionalProperties": False,
}


class NoDataError(Exception):
    """There are no transactions to give insights on."""


def build_facts(df: pd.DataFrame, budgets: list[Budget], funds: list[dict]) -> dict:
    """Everything the LLM may use. Amounts stay floats, because mask() hides long whole numbers."""
    trend = analytics.monthly_trend(df)[-MONTHS_TO_SEND:]
    months = [row["month"] for row in trend]
    last_date = df["date"].max()
    averaged = months_to_average(months, str(last_date))
    recent = [row for row in trend if row["month"] in averaged]
    monthly_income = mean(row["income"] for row in recent)
    monthly_spending = mean(row["spending"] for row in recent)
    low, high = TERM_COVER_YEARS_OF_INCOME
    return {
        "data_through": str(last_date),
        "latest_month": months[-1],
        "monthly_totals": trend,
        "budgets_latest_month": analytics.budget_status(df, budgets, months[-1]),
        "merchants": analytics.merchant_history(df, months, averaged, MERCHANTS_TO_SEND),
        "fastest_growing": analytics.fastest_growing(df, months, averaged, GROWING_TO_SEND),
        "recurring_payments": analytics.recurring_payments(df, averaged),
        "largest_payments_latest_month": analytics.largest_payments(df, months[-1], PAYMENTS_TO_SEND),
        "plan": {
            "average_monthly_income": round(monthly_income, 2),
            "average_monthly_spending": round(monthly_spending, 2),
            "average_left_over_last_3_months": round(mean(row["left_over"] for row in recent), 2),
            "emergency_fund_target": round(EMERGENCY_FUND_MONTHS * monthly_spending, 2),
            "term_cover_low": round(low * 12 * monthly_income, 2),
            "term_cover_high": round(high * 12 * monthly_income, 2),
        },
        "market": [
            {key: fund[key] for key in ("product", "fund", "return_1y_percent", "return_3y_percent_per_year", "nav_date")}
            for fund in funds
        ],
    }


def latest_month_complete(data_through: str) -> bool:
    last_date = dt.date.fromisoformat(data_through)
    return last_date.day >= calendar.monthrange(last_date.year, last_date.month)[1] - DAYS_NEAR_MONTH_END


def months_to_average(months: list[str], data_through: str) -> list[str]:
    """The last 3 whole months. A month that has only just started would pull every average down,
    which would shrink the emergency target and every saving worked out from those averages."""
    whole = months if latest_month_complete(data_through) or len(months) == 1 else months[:-1]
    return whole[-AVERAGE_OVER_MONTHS:]


def build_system(facts: dict) -> str:
    """The instructions, with only the lines that apply to these facts."""
    lines = [SYSTEM]
    if not latest_month_complete(facts["data_through"]):
        lines.append(PARTIAL_MONTH_LINE)
    if facts["plan"]["average_left_over_last_3_months"] < 0:
        lines.append(DEFICIT_LINE)
    # Left to itself, the model saw a negative figure and said spending had passed income, when it had not.
    moved_money = [
        row["month"] for row in facts["monthly_totals"] if row["left_over"] < 0 and row["spending"] <= row["income"]
    ]
    if moved_money:
        lines.append(MOVED_MONEY_LINE.format(months=", ".join(moved_money)))
    if facts["fastest_growing"]:
        lines.append(GROWING_LINE)
    if facts["recurring_payments"]:
        # Named, because told "the first one" the model picked Netflix over Cult Fit in 4 of 6 runs.
        lines.append(RECURRING_LINE.format(merchant=facts["recurring_payments"][0]["merchant"]))
    return "\n".join(lines)


def _clean_text(text) -> str:
    """Free models sometimes paste a copy of their answer into a text field, as JSON or as plain
    lines, so cut it off. A lone line break is only formatting, so it becomes a space. A sentence
    using a field's name was copied from the instructions and means nothing to the user, so it goes."""
    kept = PASTE_START.split(str(text or ""))[0]
    kept = re.sub(rf"\s*{LINE_BREAK}\s*", " ", kept).strip()
    return " ".join(sentence for sentence in SENTENCE_END.split(kept) if not FIELD_NAME.search(sentence))


def monthly_target(average: float, percent) -> float | None:
    """What is left after the LLM's percentage cut, rounded to Rs 100 so it is easy to aim for, or
    None if it is not a real percentage. The LLM gives a percentage because its rupee targets broke:
    "5,000" came back as 5 in 6 of 12 runs, which made nearly the whole average look like a saving,
    and as text it came back as ", " in 4 of 12."""
    if not isinstance(percent, (int, float)) or not 0 < percent <= 100:
        return None
    return round(average * (100 - percent) / 100, -2)


def check_cuts(answer: dict, df: pd.DataFrame, months: list[str], averaged: list[str]) -> tuple[list[dict], list]:
    """Keep cuts naming a merchant that can be cut with a saving worth having, and attach the real
    history rows.

    The saving is the merchant's recent average minus the target. Using the average and not the
    latest month means a one-off spike is not counted as money saved every month.
    """
    known = {name.lower(): name for name in analytics.cuttable_rows(df)["merchant"].unique()}

    cuts, rejected = [], []
    for cut in answer["spending_cuts"]:
        # pop, so a merchant cited twice cannot have its saving counted twice
        merchant = known.pop(str(cut.get("merchant", "")).strip().lower(), None)
        evidence = analytics.merchant_evidence(df, merchant, months, averaged) if merchant else None
        target = monthly_target(evidence["average_last_3_months"], cut.get("percent_to_cut")) if evidence else None
        if target is None or evidence["average_last_3_months"] - target < analytics.MIN_MONTHLY_CHANGE:
            rejected.append(cut.get("merchant"))
            continue
        saving = round(evidence["average_last_3_months"] - target, 2)
        cuts.append(
            {
                "merchant": merchant,
                "finding": _clean_text(cut.get("finding")),
                "action": _clean_text(cut.get("action")),
                "target_monthly_spend": target,
                "monthly_saving": saving,
                "evidence": evidence,
            }
        )
    # The prompt asks for at most six. A longer list is cut here so it cannot inflate the savings.
    return sorted(cuts, key=lambda cut: cut["monthly_saving"], reverse=True)[:MAX_CUTS], rejected


def split_available(answer: dict, available: float, funds: list[dict]) -> list[dict]:
    """Turn the LLM's percentages into rupees, rounded down to Rs 100, and add live fund figures."""
    percents = [item.get("percent_of_available") for item in answer["investment_advice"]]
    percents = [p if isinstance(p, (int, float)) and p > 0 else 0 for p in percents]
    # Shares adding up to over 100 are scaled down, so the plan can never exceed the money there is.
    whole = max(sum(percents), 100)
    fund_for = {fund["product"]: fund for fund in funds}
    return [
        {
            "product": item.get("product"),
            "percent_of_available": percent,
            "monthly_amount": math.floor(available * percent / whole / 100) * 100,
            "advice": _clean_text(item.get("advice")),
            "fund": fund_for.get(item.get("product")),
        }
        for item, percent in zip(answer["investment_advice"], percents)
    ]


def check_answer(answer: dict, df: pd.DataFrame, facts: dict, funds: list[dict]) -> dict:
    months = [row["month"] for row in facts["monthly_totals"]]
    cuts, rejected = check_cuts(answer, df, months, months_to_average(months, facts["data_through"]))
    total_saving = round(sum(cut["monthly_saving"] for cut in cuts), 2)
    # A deficit has to be covered first, so it is added before the floor at zero.
    available = max(0.0, round(facts["plan"]["average_left_over_last_3_months"] + total_saving, 2))
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "month": facts["latest_month"],
        "data_through": facts["data_through"],
        "summary": _clean_text(answer["summary"]),
        "spending_cuts": cuts,
        # Python found these, so the page can show them even when the LLM's cuts leave one out.
        "fastest_growing": facts["fastest_growing"],
        "recurring_payments": facts["recurring_payments"],
        "total_monthly_saving": total_saving,
        "available_per_month": available,
        "investment_advice": split_available(answer, available, funds),
        "plan": facts["plan"],
        "rejected_merchants": rejected,
        "disclaimer": DISCLAIMER,
    }


def generate_report(session: Session) -> dict:
    df = analytics.load_transactions(session)
    if df.empty:
        raise NoDataError("Add or import some transactions first.")

    funds = market.get_funds()
    budgets = session.exec(select(Budget)).all()
    facts = build_facts(df, budgets, funds)
    answer = ask_llm(build_system(facts), json.dumps(facts, ensure_ascii=False), SCHEMA)
    report = check_answer(answer, df, facts, funds)

    session.add(InsightReport(report=json.dumps(report)))
    session.commit()
    return report


def latest_report(session: Session) -> dict | None:
    saved = session.exec(select(InsightReport).order_by(InsightReport.id.desc())).first()
    return json.loads(saved.report) if saved else None
