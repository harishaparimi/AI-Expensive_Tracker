"""Spending numbers, worked out with pandas. The LLM never does this arithmetic."""
import pandas as pd
from sqlmodel import Session, select

from app.models import NOT_SPENDING, NOT_TO_CUT, REFUND_CATEGORIES, UNKNOWN_MERCHANT, Budget, Transaction

COLUMNS = ["id", "date", "description", "merchant", "amount", "type", "category"]
NEAR_LIMIT_PERCENT = 80
MIN_MONTHLY_CHANGE = 200  # rupees: a saving or a rise smaller than this is not worth changing a habit for
RECURRING_MONTHS = 3


def load_transactions(session: Session) -> pd.DataFrame:
    rows = session.exec(select(Transaction)).all()
    df = pd.DataFrame([row.model_dump() for row in rows], columns=COLUMNS)
    df["month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    return df


def latest_month(df: pd.DataFrame) -> str | None:
    return df["month"].max() if len(df) else None


def spending_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Money spent. A refund is a negative amount here, so it reduces what its category and merchant
    spent, everywhere these rows are used, instead of looking like income."""
    is_refund = (df["type"] == "income") & df["category"].isin(REFUND_CATEGORIES)
    rows = df[((df["type"] == "expense") & ~df["category"].isin(NOT_SPENDING)) | is_refund]
    return rows.assign(amount=rows["amount"].where(rows["type"] == "expense", -rows["amount"]))


def cuttable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The spending insights may advise cutting. The LLM only ever sees these merchants."""
    rows = spending_rows(df)
    return rows[~rows["category"].isin(NOT_TO_CUT) & (rows["merchant"] != UNKNOWN_MERCHANT)]


def _payments(amounts: pd.Series) -> int:
    """How many of these rows are payments, not refunds."""
    return int((amounts > 0).sum())


def _total(rows: pd.DataFrame) -> float:
    return round(float(rows["amount"].sum()), 2)


def month_totals(df: pd.DataFrame, month: str | None) -> dict:
    """Where a month's money went. left_over is always all money in minus all money out."""
    rows = df[df["month"] == month]
    money_in = rows[rows["type"] == "income"]
    money_out = rows[rows["type"] == "expense"]

    income = _total(money_in[~money_in["category"].isin((*NOT_SPENDING, *REFUND_CATEGORIES))])
    spending = _total(spending_rows(rows))
    invested = _total(money_out[money_out["category"] == "Investments"]) - _total(
        money_in[money_in["category"] == "Investments"]
    )
    transfers_net = _total(money_in[money_in["category"] == "Transfers"]) - _total(
        money_out[money_out["category"] == "Transfers"]
    )
    return {
        "month": month,
        "income": income,
        "spending": spending,
        "invested": round(invested, 2),
        "transfers_net": round(transfers_net, 2),
        "left_over": round(_total(money_in) - _total(money_out), 2),
        # Money invested counts as saved, so this is income minus spending.
        "savings_rate": round((income - spending) / income * 100, 1) if income else None,
    }


def monthly_trend(df: pd.DataFrame) -> list[dict]:
    return [month_totals(df, month) for month in sorted(df["month"].unique())]


def by_category(df: pd.DataFrame, month: str | None) -> list[dict]:
    rows = spending_rows(df)
    rows = rows[rows["month"] == month]
    grouped = rows.groupby("category")["amount"].agg(["sum", _payments]).sort_values("sum", ascending=False)
    # A category can net below zero when a refund lands in a later month. It has no share of spending.
    grouped = grouped[grouped["sum"] > 0]
    total = grouped["sum"].sum()
    return [
        {
            "category": category,
            "amount": round(float(row["sum"]), 2),
            "payments": int(row["_payments"]),
            "share_percent": round(float(row["sum"] / total * 100), 1),
        }
        for category, row in grouped.iterrows()
    ]


def top_merchants(df: pd.DataFrame, month: str | None, limit: int) -> list[dict]:
    rows = spending_rows(df)
    rows = rows[(rows["month"] == month) & (rows["merchant"] != UNKNOWN_MERCHANT)]
    grouped = (
        rows.groupby("merchant")
        .agg(category=("category", "first"), amount=("amount", "sum"), payments=("amount", _payments))
        .sort_values("amount", ascending=False)
    )
    grouped = grouped[grouped["amount"] > 0].head(limit)
    return [
        {
            "merchant": merchant,
            "category": row["category"],
            "amount": round(float(row["amount"]), 2),
            "payments": int(row["payments"]),
        }
        for merchant, row in grouped.iterrows()
    ]


def budget_status(df: pd.DataFrame, budgets: list[Budget], month: str | None) -> list[dict]:
    """Spent, remaining and % used for each budget, most used first."""
    rows = spending_rows(df)
    rows = rows[rows["month"] == month]
    status = []
    for budget in budgets:
        spent = _total(rows[rows["category"] == budget.category])
        percent = spent / budget.monthly_limit * 100
        status.append(
            {
                "category": budget.category,
                "monthly_limit": budget.monthly_limit,
                "spent": spent,
                "remaining": round(budget.monthly_limit - spent, 2),
                "percent_used": round(percent, 1),
                "status": "over" if percent > 100 else "near" if percent >= NEAR_LIMIT_PERCENT else "ok",
            }
        )
    return sorted(status, key=lambda item: item["percent_used"], reverse=True)


def _average_per_month(rows: pd.DataFrame, months: list[str]) -> float:
    """Divides by every month in the period, so a month with no spending counts as zero."""
    return round(_total(rows[rows["month"].isin(months)]) / len(months), 2)


def merchant_history(df: pd.DataFrame, months: list[str], averaged: list[str], limit: int) -> list[dict]:
    """Each merchant's spending per month, biggest first, so the LLM can spot patterns.
    The average is taken over the `averaged` months."""
    rows = cuttable_rows(df)
    rows = rows[rows["month"].isin(months)]
    biggest = rows.groupby("merchant")["amount"].sum().sort_values(ascending=False).head(limit)
    history = []
    for merchant in biggest.index:
        mine = rows[rows["merchant"] == merchant]
        per_month = mine.groupby("month")["amount"].sum()
        history.append(
            {
                "merchant": merchant,
                "category": mine["category"].iloc[0],
                "by_month": {month: round(float(per_month.get(month, 0.0)), 2) for month in months},
                "average_last_3_months": _average_per_month(mine, averaged),
                "payments_latest_month": _payments(mine[mine["month"] == months[-1]]["amount"]),
            }
        )
    return history


def recurring_payments(df: pd.DataFrame, averaged: list[str]) -> list[dict]:
    """Merchants charging the same amount exactly once in each of the last 3 whole months, such as a
    membership. A small monthly charge is easy to overlook, so the yearly cost is what shows it."""
    if len(averaged) < RECURRING_MONTHS:
        return []
    rows = cuttable_rows(df)
    rows = rows[rows["month"].isin(averaged) & (rows["amount"] > 0)]
    found = [
        {"merchant": merchant, "monthly_amount": float(amount), "yearly_cost": round(float(amount) * 12, 2)}
        for (merchant, amount), charges in rows.groupby(["merchant", "amount"])
        if sorted(charges["month"]) == averaged
    ]
    return sorted(found, key=lambda item: item["yearly_cost"], reverse=True)


def fastest_growing(df: pd.DataFrame, months: list[str], averaged: list[str], limit: int) -> list[dict]:
    """Merchants whose monthly spending rose most: the average of the months before the `averaged`
    months against the average of those months. A big one-off payment can also put a merchant here."""
    earlier = [month for month in months if month < averaged[0]]
    if not earlier:
        return []
    rising = []
    for merchant, mine in cuttable_rows(df).groupby("merchant"):
        before = _average_per_month(mine, earlier)
        recent = _average_per_month(mine, averaged)
        if recent - before >= MIN_MONTHLY_CHANGE:
            rising.append(
                {
                    "merchant": merchant,
                    "earlier_average": before,
                    "recent_average": recent,
                    "rise_per_month": round(recent - before, 2),
                }
            )
    return sorted(rising, key=lambda item: item["rise_per_month"], reverse=True)[:limit]


def largest_payments(df: pd.DataFrame, month: str, limit: int) -> list[dict]:
    rows = cuttable_rows(df)
    rows = rows[(rows["month"] == month) & (rows["amount"] > 0)].nlargest(limit, "amount")
    return [
        {"date": str(row["date"]), "merchant": row["merchant"], "amount": float(row["amount"])}
        for _, row in rows.iterrows()
    ]


def merchant_evidence(df: pd.DataFrame, merchant: str, months: list[str], averaged: list[str], limit: int = 3) -> dict:
    """The real figures and history rows behind an insight, taken from the data and not the LLM."""
    rows = cuttable_rows(df)
    rows = rows[rows["merchant"] == merchant]
    newest = rows.sort_values("date", ascending=False).head(limit)
    return {
        "category": newest["category"].iloc[0],
        "latest_month_spent": _total(rows[rows["month"] == months[-1]]),
        "average_last_3_months": _average_per_month(rows, averaged),
        "recent_payments": [
            {"date": str(row["date"]), "description": row["description"], "amount": float(row["amount"])}
            for _, row in newest.iterrows()
        ],
    }
