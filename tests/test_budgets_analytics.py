from urllib.parse import quote

import pytest

FOOD = quote("Food & Dining")


def add(client, date, amount, category, type="expense", description="Payment"):
    response = client.post(
        "/transactions",
        json={"date": date, "description": description, "amount": amount, "type": type, "category": category},
    )
    assert response.status_code == 201, response.text


@pytest.fixture
def history(client, fake_llm):
    """Two months, with money moving in every way that should not count as spending."""
    add(client, "2026-09-05", 1000, "Food & Dining", description="Swiggy")
    add(client, "2026-09-30", 50000, "Income", type="income", description="Salary")
    add(client, "2026-10-01", 60000, "Income", type="income", description="Salary")
    add(client, "2026-10-02", 18000, "Rent & Housing", description="Rent")
    add(client, "2026-10-03", 3000, "Food & Dining", description="Swiggy")
    add(client, "2026-10-04", 1500.5, "Food & Dining", description="Zomato")
    add(client, "2026-10-05", 5000, "Investments", description="SIP")
    add(client, "2026-10-06", 2000, "Transfers", description="Sent to a friend")
    add(client, "2026-10-07", 500, "Transfers", type="income", description="Friend repaid")
    return client


def test_a_refund_reduces_spending_and_is_not_income(client, fake_llm):
    add(client, "2026-10-01", 50000, "Income", type="income", description="Salary")
    add(client, "2026-10-03", 5000, "Shopping", description="Myntra")
    add(client, "2026-10-05", 1500, "Shopping", type="income", description="Myntra")

    summary = client.get("/analytics/summary").json()
    assert summary["income"] == 50000.0
    assert summary["spending"] == 3500.0
    assert summary["left_over"] == 46500.0  # still all money in minus all money out
    assert summary["savings_rate"] == 93.0


def test_a_refund_reduces_its_category_merchant_and_budget(client, fake_llm):
    add(client, "2026-10-03", 5000, "Shopping", description="Myntra")
    add(client, "2026-10-05", 1500, "Shopping", type="income", description="Myntra")
    client.put("/budgets/Shopping", json={"monthly_limit": 4000})

    assert client.get("/analytics/categories").json()["categories"] == [
        {"category": "Shopping", "amount": 3500.0, "payments": 1, "share_percent": 100.0}
    ]
    merchant = client.get("/analytics/merchants").json()["merchants"][0]
    assert (merchant["amount"], merchant["payments"]) == (3500.0, 1)
    budget = client.get("/budgets").json()["budgets"][0]
    assert (budget["spent"], budget["remaining"], budget["status"]) == (3500.0, 500.0, "near")


def test_a_refund_in_a_later_month_gets_no_share_of_that_months_spending(client, fake_llm):
    add(client, "2026-09-03", 5000, "Shopping", description="Myntra")
    add(client, "2026-10-05", 1500, "Shopping", type="income", description="Myntra")
    add(client, "2026-10-06", 1000, "Groceries", description="Blinkit")

    assert client.get("/analytics/summary").json()["spending"] == -500.0
    assert client.get("/analytics/categories").json()["categories"] == [
        {"category": "Groceries", "amount": 1000.0, "payments": 1, "share_percent": 100.0}
    ]
    assert [m["merchant"] for m in client.get("/analytics/merchants").json()["merchants"]] == ["Blinkit"]


@pytest.mark.parametrize("category", ["Income", "Other"])
def test_money_in_that_is_not_a_refund_stays_income(client, fake_llm, category):
    add(client, "2026-10-01", 5000, category, type="income", description="Money in")
    add(client, "2026-10-02", 200, "Shopping", description="Myntra")

    summary = client.get("/analytics/summary").json()
    assert (summary["income"], summary["spending"]) == (5000.0, 200.0)


def test_uncategorized_money_in_stays_income_so_a_failed_llm_cannot_erase_a_salary(client, fake_llm):
    fake_llm.fail = True
    response = client.post(
        "/transactions", json={"date": "2026-10-01", "description": "SALARY", "amount": 50000, "type": "income"}
    )
    assert response.json()["category"] == "Uncategorized"

    summary = client.get("/analytics/summary").json()
    assert (summary["income"], summary["spending"]) == (50000.0, 0.0)


def test_summary_keeps_transfers_and_investments_out_of_spending(history):
    summary = history.get("/analytics/summary").json()
    assert summary == {
        "month": "2026-10",  # the latest month with data
        "income": 60000.0,
        "spending": 22500.5,
        "invested": 5000.0,
        "transfers_net": -1500.0,
        "left_over": 30999.5,
        "savings_rate": 62.5,
    }


def test_left_over_ties_back_to_all_money_in_minus_all_money_out(history):
    s = history.get("/analytics/summary").json()
    assert s["left_over"] == s["income"] - s["spending"] - s["invested"] + s["transfers_net"]
    rows = history.get("/transactions", params={"start": "2026-10-01"}).json()
    assert s["left_over"] == sum(r["amount"] if r["type"] == "income" else -r["amount"] for r in rows)


def test_month_filter_and_trend(history):
    assert history.get("/analytics/summary", params={"month": "2026-09"}).json()["spending"] == 1000.0
    assert [row["month"] for row in history.get("/analytics/trend").json()] == ["2026-09", "2026-10"]


def test_categories_and_merchants_are_spending_only_and_sorted(history):
    categories = history.get("/analytics/categories").json()["categories"]
    assert [c["category"] for c in categories] == ["Rent & Housing", "Food & Dining"]
    assert categories[1]["amount"] == 4500.5
    assert categories[1]["payments"] == 2
    assert sum(c["share_percent"] for c in categories) == pytest.approx(100, abs=0.2)

    merchants = history.get("/analytics/merchants", params={"limit": 2}).json()["merchants"]
    assert [m["merchant"] for m in merchants] == ["Rent", "Swiggy"]


def test_analytics_and_budgets_work_with_no_data(client):
    assert client.get("/analytics/summary").json()["income"] == 0
    assert client.get("/analytics/categories").json() == {"month": None, "categories": []}
    assert client.get("/analytics/merchants").json() == {"month": None, "merchants": []}
    assert client.get("/analytics/trend").json() == []
    assert client.get("/budgets").json() == {"month": None, "budgets": []}


def test_bad_month_is_rejected(client):
    assert client.get("/analytics/summary", params={"month": "2026-13"}).status_code == 422
    assert client.get("/budgets", params={"month": "October"}).status_code == 422


def test_budget_status_defaults_to_latest_month(history):
    assert history.put(f"/budgets/{FOOD}", json={"monthly_limit": 4000}).status_code == 200
    assert history.put("/budgets/Transport", json={"monthly_limit": 1000}).status_code == 200

    status = history.get("/budgets").json()
    assert status["month"] == "2026-10"
    food, transport = status["budgets"]  # most used first
    assert food == {
        "category": "Food & Dining",
        "monthly_limit": 4000.0,
        "spent": 4500.5,
        "remaining": -500.5,
        "percent_used": 112.5,
        "status": "over",
    }
    assert (transport["spent"], transport["status"]) == (0.0, "ok")

    september = history.get("/budgets", params={"month": "2026-09"}).json()["budgets"]
    assert next(b for b in september if b["category"] == "Food & Dining")["spent"] == 1000.0


@pytest.mark.parametrize("limit, expected", [(5625, "near"), (4501, "near"), (5626, "ok"), (4500, "over"), (4500.25, "over")])
def test_budget_status_thresholds(history, limit, expected):
    history.put(f"/budgets/{FOOD}", json={"monthly_limit": limit})
    assert history.get("/budgets").json()["budgets"][0]["status"] == expected


def test_setting_a_budget_again_replaces_it(client):
    client.put(f"/budgets/{FOOD}", json={"monthly_limit": 4000})
    client.put(f"/budgets/{FOOD}", json={"monthly_limit": 6000})
    budgets = client.get("/budgets").json()["budgets"]
    assert [b["monthly_limit"] for b in budgets] == [6000.0]


@pytest.mark.parametrize(
    "category, limit",
    [("Income", 100), ("Investments", 100), ("Transfers", 100), ("Made Up", 100), ("Transport", 0), ("Transport", -5)],
)
def test_invalid_budgets_are_rejected(client, category, limit):
    assert client.put(f"/budgets/{category}", json={"monthly_limit": limit}).status_code == 422


def test_delete_budget(client):
    client.put(f"/budgets/{FOOD}", json={"monthly_limit": 4000})
    assert client.delete(f"/budgets/{FOOD}").status_code == 204
    assert client.delete(f"/budgets/{FOOD}").status_code == 404
    assert client.get("/budgets").json()["budgets"] == []
