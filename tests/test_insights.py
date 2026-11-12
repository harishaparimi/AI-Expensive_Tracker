import json

import pytest
import requests

from app.services import analytics, insights, market
from app.services.llm import mask


def fund_data(name="Test Fund"):
    """A fund priced 100 three years ago, 150 a year ago and 165 now."""
    return {
        "meta": {"scheme_name": name},
        "data": [
            {"date": "04-11-2026", "nav": "165.0"},
            {"date": "03-11-2025", "nav": "150.0"},  # the date a year back has no price
            {"date": "01-11-2023", "nav": "100.0"},
            {"date": "01-01-2023", "nav": "90.0"},
        ],
    }


@pytest.fixture
def live_market(monkeypatch):
    """Stands in for mfapi.in and counts the requests."""
    calls = []

    def fetch(code):
        calls.append(code)
        return fund_data()

    monkeypatch.setattr(market, "fetch_fund", fetch)
    return calls


@pytest.fixture
def market_down(monkeypatch):
    def fetch(code):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(market, "fetch_fund", fetch)


def test_returns_use_the_last_price_on_or_before_the_date():
    fund = market.summarize_fund(118482, fund_data())
    assert fund["nav"] == 165.0
    assert fund["nav_date"] == "2026-11-04"
    assert fund["product"] == "Equity mutual fund"
    assert fund["return_1y_percent"] == 10.0
    assert fund["return_3y_percent_per_year"] == pytest.approx(18.17, abs=0.01)  # 100 to 165 in 3 years


def test_a_fund_younger_than_the_period_has_no_return():
    data = fund_data()
    data["data"] = data["data"][:2]
    fund = market.summarize_fund(118482, data)
    assert fund["return_1y_percent"] == 10.0
    assert fund["return_3y_percent_per_year"] is None


def test_market_is_fetched_once_a_day(client, live_market):
    first = client.get("/market").json()
    assert len(live_market) == 3
    assert client.get("/market").json() == first
    assert len(live_market) == 3
    assert all(fund["stale"] is False for fund in first)


def test_market_falls_back_to_the_last_saved_copy(client, live_market, temp_db, monkeypatch):
    client.get("/market")
    with temp_db.begin() as connection:  # pretend that was fetched on an earlier day
        connection.exec_driver_sql("UPDATE fundsnapshot SET fetched_on = '2026-01-01'")

    def down(code):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(market, "fetch_fund", down)
    funds = client.get("/market").json()
    assert len(funds) == 3
    assert all(fund["stale"] is True and fund["nav"] == 165.0 for fund in funds)


def test_market_with_no_saved_copy_leaves_funds_out(client, market_down):
    assert client.get("/market").json() == []


def add(client, date, amount, category, description, type="expense"):
    response = client.post(
        "/transactions",
        json={"date": date, "description": description, "amount": amount, "type": type, "category": category},
    )
    assert response.status_code == 201, response.text


@pytest.fixture
def history(client, fake_llm, live_market):
    add(client, "2026-09-01", 200000, "Income", "Salary", type="income")
    add(client, "2026-10-01", 200000, "Income", "Salary", type="income")
    add(client, "2026-09-10", 1000, "Food & Dining", "Swiggy")
    add(client, "2026-10-10", 3000, "Food & Dining", "Swiggy")
    add(client, "2026-10-12", 500, "Food & Dining", "Swiggy")
    add(client, "2026-10-15", 150000, "Rent & Housing", "Landlord")
    add(client, "2026-10-20", 7000, "Investments", "Groww")
    add(client, "2026-10-31", 2000, "Transfers", "Friend")
    client.put("/budgets/Shopping", json={"monthly_limit": 123456})
    fake_llm.insights = {
        "summary": "Spending rose.",
        "spending_cuts": [
            {"merchant": "swiggy", "finding": "Rising.", "action": "Order less.", "percent_to_cut": 60},
            {"merchant": "Made Up Cafe", "finding": "Invented.", "action": "Stop.", "percent_to_cut": 100},
        ],
        "investment_advice": [
            {"product": "Equity mutual fund", "percent_of_available": 60, "advice": "Index funds."},
            {"product": "Term insurance", "percent_of_available": 0, "advice": "Compare quotes."},
        ],
    }
    return client


def sent_facts(fake_llm):
    return json.loads(fake_llm.calls[-1])


def test_cuts_naming_unknown_merchants_are_removed(history, fake_llm):
    report = history.post("/insights").json()
    assert [cut["merchant"] for cut in report["spending_cuts"]] == ["Swiggy"]  # spelled as in the data
    assert report["rejected_merchants"] == ["Made Up Cafe"]
    assert report["total_monthly_saving"] == 1350  # average 2250 minus target 900; the invented cut adds nothing


def test_cuts_carry_real_figures_and_history_rows(history, fake_llm):
    evidence = history.post("/insights").json()["spending_cuts"][0]["evidence"]
    assert evidence["category"] == "Food & Dining"
    assert evidence["latest_month_spent"] == 3500.0
    assert evidence["average_last_3_months"] == 2250.0
    assert [(p["date"], p["amount"]) for p in evidence["recent_payments"]] == [
        ("2026-10-12", 500.0),
        ("2026-10-10", 3000.0),
        ("2026-09-10", 1000.0),
    ]


def test_a_one_off_spike_is_not_counted_as_a_saving_every_month(client, fake_llm, live_market):
    """Nothing in August, 300 in September and 6000 in October: the average includes the empty month."""
    add(client, "2026-08-01", 100, "Food & Dining", "Swiggy")
    add(client, "2026-09-01", 300, "Shopping", "Myntra")
    add(client, "2026-10-31", 6000, "Shopping", "Myntra")
    fake_llm.insights = {
        "summary": "",
        "spending_cuts": [{"merchant": "Myntra", "finding": "", "action": "", "percent_to_cut": 60}],
        "investment_advice": [],
    }
    cut = client.post("/insights").json()["spending_cuts"][0]
    assert cut["evidence"]["average_last_3_months"] == 2100.0
    assert cut["target_monthly_spend"] == 800.0
    assert cut["monthly_saving"] == 1300.0  # not 5200, which measures against the October spike


@pytest.mark.parametrize("percent", [0, 101, -5, "a lot", None])
def test_a_percentage_that_is_not_real_is_rejected(history, fake_llm, percent):
    fake_llm.insights = {
        **fake_llm.insights,
        "spending_cuts": [{"merchant": "Swiggy", "finding": "", "action": "", "percent_to_cut": percent}],
    }
    report = history.post("/insights").json()
    assert report["spending_cuts"] == []
    assert report["total_monthly_saving"] == 0


@pytest.mark.parametrize(
    "average, percent, target",
    [
        (4924.33, 50, 2500.0),  # 2462.17, rounded to Rs 100
        (1499.0, 100, 0.0),
        (5405.33, 1, 5400.0),
        (1000.0, 0, None),
        (1000.0, 100.5, None),
        (1000.0, "half", None),
    ],
)
def test_python_works_out_the_target_from_the_percentage(average, percent, target):
    assert insights.monthly_target(average, percent) == target


def test_a_merchant_cited_twice_is_counted_once(history, fake_llm):
    cut = {"merchant": "Swiggy", "finding": "", "action": "", "percent_to_cut": 60}
    fake_llm.insights = {**fake_llm.insights, "spending_cuts": [cut, {**cut, "merchant": "SWIGGY"}]}
    report = history.post("/insights").json()
    assert len(report["spending_cuts"]) == 1
    assert report["total_monthly_saving"] == 1350
    assert report["rejected_merchants"] == ["SWIGGY"]


def test_cuts_are_sorted_by_the_computed_saving(history, fake_llm):
    add(history, "2026-10-14", 9000, "Shopping", "Myntra")
    fake_llm.insights = {
        **fake_llm.insights,
        "spending_cuts": [
            {"merchant": "Swiggy", "finding": "", "action": "", "percent_to_cut": 20},
            {"merchant": "Myntra", "finding": "", "action": "", "percent_to_cut": 50},
        ],
    }
    assert [cut["merchant"] for cut in history.post("/insights").json()["spending_cuts"]] == ["Myntra", "Swiggy"]


@pytest.mark.parametrize("category", ["Rent & Housing", "Bills & Utilities", "Health", "Other"])
def test_needs_and_cash_cannot_be_cited_as_a_spending_cut(client, fake_llm, live_market, category):
    add(client, "2026-10-10", 5000, category, "Some Place")
    fake_llm.insights = {
        "summary": "",
        "spending_cuts": [{"merchant": "Some Place", "finding": "", "action": "", "percent_to_cut": 100}],
        "investment_advice": [],
    }
    report = client.post("/insights").json()
    assert report["spending_cuts"] == []
    assert report["rejected_merchants"] == ["Some Place"]
    assert report["total_monthly_saving"] == 0
    facts = sent_facts(fake_llm)
    assert facts["merchants"] == []
    assert facts["largest_payments_latest_month"] == []


def test_a_merchant_in_two_categories_is_only_judged_on_what_can_be_cut(client, fake_llm, live_market):
    add(client, "2026-10-10", 4000, "Shopping", "Amazon")
    add(client, "2026-10-11", 1000, "Health", "Amazon")
    fake_llm.insights = {
        "summary": "",
        "spending_cuts": [{"merchant": "Amazon", "finding": "", "action": "", "percent_to_cut": 25}],
        "investment_advice": [],
    }
    cut = client.post("/insights").json()["spending_cuts"][0]
    assert cut["monthly_saving"] == 1000.0
    assert [p["amount"] for p in cut["evidence"]["recent_payments"]] == [4000.0]
    assert sent_facts(fake_llm)["merchants"][0]["average_last_3_months"] == 4000.0


@pytest.mark.parametrize("percent, kept", [(20, True), (14, False)])
def test_a_saving_too_small_to_matter_is_rejected(client, fake_llm, live_market, percent, kept):
    """Averaging 1000, a 20 percent cut saves exactly the minimum, and 14 percent rounds to Rs 100 saved."""
    add(client, "2026-10-10", 1000, "Shopping", "Myntra")
    fake_llm.insights = {
        "summary": "",
        "spending_cuts": [{"merchant": "Myntra", "finding": "", "action": "", "percent_to_cut": percent}],
        "investment_advice": [],
    }
    report = client.post("/insights").json()
    assert [cut["merchant"] for cut in report["spending_cuts"]] == (["Myntra"] if kept else [])
    assert report["total_monthly_saving"] == (analytics.MIN_MONTHLY_CHANGE if kept else 0)


def monthly(client, merchant, category, amounts, day=10):
    """One payment a month from May onward. A zero means no payment that month."""
    for number, amount in enumerate(amounts, start=5):
        if amount:
            add(client, f"2026-{number:02d}-{day:02d}", amount, category, merchant)


def patterns(client):
    """A report on May to October, with a salary so every month has a row, as a real statement
    would. The last 3 whole months are August to October."""
    for number in range(5, 11):
        add(client, f"2026-{number:02d}-28", 50000, "Income", "Salary", type="income")
    return client.post("/insights").json()


def test_a_membership_is_found_with_its_yearly_cost(client, fake_llm, live_market):
    monthly(client, "Cult Fit", "Subscriptions", [1499] * 6)
    monthly(client, "Spotify", "Subscriptions", [119] * 6)
    report = patterns(client)
    assert report["recurring_payments"] == [
        {"merchant": "Cult Fit", "monthly_amount": 1499.0, "yearly_cost": 17988.0},
        {"merchant": "Spotify", "monthly_amount": 119.0, "yearly_cost": 1428.0},
    ]
    assert sent_facts(fake_llm)["recurring_payments"] == report["recurring_payments"]
    assert insights.RECURRING_LINE.format(merchant="Cult Fit") in fake_llm.systems[-1]  # the costliest, by name


@pytest.mark.parametrize(
    "merchant, category, amounts",
    [
        ("Bescom", "Food & Dining", [900, 1000, 1100, 1200, 1300, 1400]),  # the amount changes
        ("Old Gym", "Subscriptions", [999, 999, 999, 999, 999, 0]),  # stopped
        ("Swiggy", "Food & Dining", [349, 0, 0, 349, 0, 349]),  # the same amount by chance
        ("Landlord", "Rent & Housing", [18000] * 6),  # never a cut
        ("ATM", "Other", [2000] * 6),
    ],
)
def test_other_payments_are_not_called_recurring(client, fake_llm, live_market, merchant, category, amounts):
    monthly(client, merchant, category, amounts)
    assert patterns(client)["recurring_payments"] == []
    assert "recurring_payments lists" not in fake_llm.systems[-1]


def test_paying_the_same_amount_twice_a_month_is_not_a_membership(client, fake_llm, live_market):
    monthly(client, "Rapido", "Transport", [80] * 6, day=10)
    monthly(client, "Rapido", "Transport", [80] * 6, day=20)
    assert patterns(client)["recurring_payments"] == []


def test_recurring_needs_three_whole_months(client, fake_llm, live_market):
    add(client, "2026-09-10", 1499, "Subscriptions", "Cult Fit")
    add(client, "2026-10-10", 1499, "Subscriptions", "Cult Fit")
    assert client.post("/insights").json()["recurring_payments"] == []


def test_growing_merchants_compare_the_earlier_months_with_the_last_three(client, fake_llm, live_market):
    monthly(client, "Swiggy", "Food & Dining", [1000, 1000, 1000, 2000, 3000, 4000])
    monthly(client, "Third Wave", "Food & Dining", [500, 500, 500, 1000, 1000, 1000])
    monthly(client, "Blinkit", "Groceries", [1000, 1000, 1000, 1100, 1100, 1100])  # too small a rise
    monthly(client, "Uber", "Transport", [3000, 3000, 3000, 1000, 1000, 1000])  # falling
    monthly(client, "Apollo", "Health", [100, 100, 100, 5000, 5000, 5000])  # never a cut
    report = patterns(client)
    assert report["fastest_growing"] == [
        {"merchant": "Swiggy", "earlier_average": 1000.0, "recent_average": 3000.0, "rise_per_month": 2000.0},
        {"merchant": "Third Wave", "earlier_average": 500.0, "recent_average": 1000.0, "rise_per_month": 500.0},
    ]
    assert sent_facts(fake_llm)["fastest_growing"] == report["fastest_growing"]
    assert insights.GROWING_LINE in fake_llm.systems[-1]


def test_only_the_fastest_growing_are_sent(client, fake_llm, live_market):
    for number in range(1, insights.GROWING_TO_SEND + 2):
        add(client, "2026-05-10", 100, "Shopping", f"Shop {number}")
        add(client, "2026-10-10", 3000 * number, "Shopping", f"Shop {number}")
    growing = patterns(client)["fastest_growing"]
    assert len(growing) == insights.GROWING_TO_SEND
    assert growing[0]["merchant"] == f"Shop {insights.GROWING_TO_SEND + 1}"


def test_nothing_is_growing_without_earlier_months(client, fake_llm, live_market):
    add(client, "2026-10-10", 5000, "Shopping", "Myntra")
    assert client.post("/insights").json()["fastest_growing"] == []
    assert insights.GROWING_LINE not in fake_llm.systems[-1]


def test_a_part_month_is_not_used_to_find_patterns(client, fake_llm, live_market):
    """The statement stops on 5 October, before the gym is charged, so it must still count."""
    monthly(client, "Cult Fit", "Subscriptions", [1499] * 5)
    add(client, "2026-10-05", 100, "Food & Dining", "Swiggy")
    assert [item["merchant"] for item in client.post("/insights").json()["recurring_payments"]] == ["Cult Fit"]


def test_the_instructions_do_not_force_a_number_of_cuts():
    assert "four to six" not in insights.SYSTEM


def test_only_the_six_largest_cuts_count_toward_the_savings(client, fake_llm, live_market):
    spent = [100, 200, 300, 400, 500, 600, 700, 800]
    for amount in spent:
        add(client, "2026-10-10", amount, "Shopping", f"Shop {amount}")
    fake_llm.insights = {
        "summary": "",
        "spending_cuts": [
            {"merchant": f"Shop {amount}", "finding": "", "action": "", "percent_to_cut": 100} for amount in spent
        ],
        "investment_advice": [],
    }
    report = client.post("/insights").json()
    assert [cut["merchant"] for cut in report["spending_cuts"]] == [f"Shop {n}" for n in (800, 700, 600, 500, 400, 300)]
    assert report["total_monthly_saving"] == 3300.0  # the two smallest are left out


def spending_and_split(client, fake_llm, income, spent, percent_to_cut, percents):
    add(client, "2026-10-01", spent, "Food & Dining", "Swiggy")
    add(client, "2026-10-01", income, "Income", "Salary", type="income")
    products = ["Emergency fund", "Term insurance", "Equity mutual fund", "Gold"]
    fake_llm.insights = {
        "summary": "",
        "spending_cuts": [{"merchant": "Swiggy", "finding": "", "action": "", "percent_to_cut": percent_to_cut}],
        "investment_advice": [
            {"product": product, "percent_of_available": percent, "advice": ""}
            for product, percent in zip(products, percents)
        ],
    }
    return client.post("/insights").json()


def amounts(report):
    return [item["monthly_amount"] for item in report["investment_advice"]]


def test_available_money_is_the_left_over_plus_the_savings(client, fake_llm, live_market):
    report = spending_and_split(client, fake_llm, 10000, 5000, 20, [60, 0, 30, 10])
    assert report["available_per_month"] == 6000.0  # left over 5000 plus saving 1000
    assert amounts(report) == [3600, 0, 1800, 600]


def test_a_deficit_is_covered_before_anything_is_invested(client, fake_llm, live_market):
    report = spending_and_split(client, fake_llm, 1000, 5000, 20, [60, 0, 30, 10])
    assert report["available_per_month"] == 0  # left over -4000 plus saving 1000 is still negative
    assert amounts(report) == [0, 0, 0, 0]


def test_a_split_over_100_percent_cannot_exceed_the_money_available(client, fake_llm, live_market):
    report = spending_and_split(client, fake_llm, 10000, 5000, 20, [100, 0, 100, 0])
    assert report["available_per_month"] == 6000.0
    assert amounts(report) == [3000, 0, 3000, 0]
    assert sum(amounts(report)) <= report["available_per_month"]


# The first three are the shapes real free-model answers came back in.
@pytest.mark.parametrize(
    "paste",
    [
        ' \n\n{"summary": "Spending rose.", "spending_cuts": []}',  # JSON after a blank line
        '\\n{"summary": "Spending rose.", "spending_cuts": []}',  # a line break written out as backslash-n
        "\n\nspending_cuts:\n- merchant: SWIGGY\n- finding: Rising.\n",  # a plain text copy
        "\r\n\r\nspending_cuts:\r\n- merchant: SWIGGY",  # the same with Windows line endings
        "\\n\\nspending_cuts:\\n- merchant: SWIGGY",  # the same with line breaks written out
        '\n{ "summary": "Spending rose." }',  # JSON with a space after the brace
    ],
)
def test_a_copy_of_the_answer_pasted_into_text_is_removed(history, fake_llm, paste):
    fake_llm.insights = {
        "summary": "Spending rose." + paste,
        "spending_cuts": [
            {"merchant": "Swiggy", "finding": "Rising." + paste, "action": "Cap it." + paste, "percent_to_cut": 50}
        ],
        "investment_advice": [{"product": "Gold", "percent_of_available": 10, "advice": "Small share." + paste}],
    }
    report = history.post("/insights").json()
    assert report["summary"] == "Spending rose."
    assert report["spending_cuts"][0]["finding"] == "Rising."
    assert report["spending_cuts"][0]["action"] == "Cap it."
    assert report["investment_advice"][0]["advice"] == "Small share."


@pytest.mark.parametrize("line_break", ["\n", "\\n", "\r\n", " \n "])
def test_a_single_line_break_keeps_the_text_after_it(history, fake_llm, line_break):
    """A lone line break is only formatting, so a real second sentence must not be lost."""
    fake_llm.insights = {**fake_llm.insights, "summary": f"Spending rose.{line_break}Savings fell."}
    assert history.post("/insights").json()["summary"] == "Spending rose. Savings fell."


def test_money_moving_cannot_be_cited_as_a_spending_cut(history, fake_llm):
    fake_llm.insights = {
        **fake_llm.insights,
        "spending_cuts": [{"merchant": "Groww", "finding": "", "action": "", "percent_to_cut": 100}],
    }
    report = history.post("/insights").json()
    assert report["spending_cuts"] == []
    assert report["rejected_merchants"] == ["Groww"]


def test_advice_carries_live_fund_figures_and_a_disclaimer(history, fake_llm):
    report = history.post("/insights").json()
    equity, term = report["investment_advice"]
    assert equity["fund"]["return_1y_percent"] == 10.0
    assert term["fund"] is None
    assert "SEBI" in report["disclaimer"]


def test_facts_sent_to_the_llm(history, fake_llm):
    history.post("/insights")
    facts = sent_facts(fake_llm)
    assert facts["latest_month"] == "2026-10"
    assert facts["data_through"] == "2026-10-31"
    assert {m["merchant"] for m in facts["merchants"]} == {"Swiggy"}  # rent is never offered as a cut
    swiggy = next(m for m in facts["merchants"] if m["merchant"] == "Swiggy")
    assert swiggy["by_month"] == {"2026-09": 1000.0, "2026-10": 3500.0}
    assert swiggy["average_last_3_months"] == 2250.0
    assert swiggy["payments_latest_month"] == 2
    assert facts["largest_payments_latest_month"][0] == {"date": "2026-10-10", "merchant": "Swiggy", "amount": 3000.0}
    assert facts["budgets_latest_month"][0]["monthly_limit"] == 123456.0

    plan = facts["plan"]
    assert plan["average_monthly_income"] == 200000.0
    assert plan["average_monthly_spending"] == 77250.0
    assert plan["emergency_fund_target"] == 463500.0
    assert plan["term_cover_low"] == 24000000.0
    assert plan["term_cover_high"] == 36000000.0


@pytest.mark.parametrize(
    "last_date, complete",
    [("2026-10-31", True), ("2026-10-28", True), ("2026-10-27", False), ("2026-10-05", False), ("2026-09-30", True), ("2026-02-25", True)],
)
def test_python_decides_if_the_latest_month_is_complete(client, fake_llm, live_market, last_date, complete):
    add(client, last_date, 500, "Shopping", "Myntra")
    client.post("/insights")
    assert (insights.PARTIAL_MONTH_LINE not in fake_llm.systems[-1]) is complete
    assert "complete" not in fake_llm.calls[-1]


@pytest.mark.parametrize(
    "months, last, expected",
    [
        (["2026-07", "2026-08", "2026-09", "2026-10"], "2026-10-31", ["2026-08", "2026-09", "2026-10"]),
        (["2026-07", "2026-08", "2026-09", "2026-10"], "2026-10-20", ["2026-07", "2026-08", "2026-09"]),
        (["2026-09", "2026-10"], "2026-10-05", ["2026-09"]),
        (["2026-10"], "2026-10-05", ["2026-10"]),  # nothing whole to use, so the part month has to do
    ],
)
def test_averages_use_whole_months(months, last, expected):
    assert insights.months_to_average(months, last) == expected


def test_a_month_that_has_just_started_does_not_lower_the_plan_or_the_savings(client, fake_llm, live_market):
    for month in ("2026-07", "2026-08", "2026-09"):
        add(client, f"{month}-01", 50000, "Income", "Salary", type="income")
        add(client, f"{month}-10", 3000, "Food & Dining", "Swiggy")
    add(client, "2026-10-02", 100, "Food & Dining", "Swiggy")  # the statement stops here
    fake_llm.insights = {
        "summary": "",
        "spending_cuts": [{"merchant": "Swiggy", "finding": "", "action": "", "percent_to_cut": 50}],
        "investment_advice": [],
    }
    report = client.post("/insights").json()

    assert report["plan"]["average_monthly_spending"] == 3000.0  # 2,033 with the part month counted
    assert report["plan"]["average_monthly_income"] == 50000.0
    assert report["plan"]["emergency_fund_target"] == 18000.0
    cut = report["spending_cuts"][0]
    assert cut["evidence"]["average_last_3_months"] == 3000.0
    assert cut["monthly_saving"] == 1500.0
    assert cut["evidence"]["latest_month_spent"] == 100.0  # the latest month is still shown as it is
    assert sent_facts(fake_llm)["merchants"][0]["average_last_3_months"] == 3000.0


def test_a_refund_lowers_the_merchants_figures_and_the_saving(client, fake_llm, live_market):
    add(client, "2026-10-05", 4000, "Shopping", "Myntra")
    add(client, "2026-10-31", 1000, "Shopping", "Myntra", type="income")
    fake_llm.insights = {
        "summary": "",
        "spending_cuts": [{"merchant": "Myntra", "finding": "", "action": "", "percent_to_cut": 50}],
        "investment_advice": [],
    }
    report = client.post("/insights").json()

    facts = sent_facts(fake_llm)["merchants"][0]
    assert facts["by_month"]["2026-10"] == 3000.0
    assert facts["payments_latest_month"] == 1
    cut = report["spending_cuts"][0]
    assert cut["evidence"]["latest_month_spent"] == 3000.0
    assert cut["monthly_saving"] == 1500.0
    assert [p["amount"] for p in cut["evidence"]["recent_payments"]] == [-1000.0, 4000.0]


@pytest.mark.parametrize(
    "income, spent, invested, applies",
    [
        (65000, 63000, 3000, True),  # spending is below income, and investing took it negative
        (65000, 70000, 3000, False),  # spending itself went above income
        (65000, 50000, 3000, False),  # nothing went negative
    ],
)
def test_python_says_why_money_left_over_went_negative(client, fake_llm, live_market, income, spent, invested, applies):
    add(client, "2026-10-01", income, "Income", "Salary", type="income")
    add(client, "2026-10-10", spent, "Shopping", "Myntra")
    add(client, "2026-10-20", invested, "Investments", "Groww")
    client.post("/insights")
    assert (insights.MOVED_MONEY_LINE.format(months="2026-10") in fake_llm.systems[-1]) is applies


def test_the_deficit_line_is_sent_only_when_money_ran_short_on_average(client, fake_llm, live_market):
    add(client, "2026-09-30", 1000, "Income", "Salary", type="income")
    add(client, "2026-10-31", 500, "Shopping", "Myntra")
    client.post("/insights")
    assert insights.DEFICIT_LINE not in fake_llm.systems[-1]

    add(client, "2026-10-30", 5000, "Shopping", "Myntra")
    client.post("/insights")
    assert insights.DEFICIT_LINE in fake_llm.systems[-1]


@pytest.mark.parametrize(
    "text, kept",
    [
        ("Build it first. You invest Rs 3,000 in monthly_totals, so it is on top. Keep it liquid.", "Build it first. Keep it liquid."),
        ("left_over is all money in minus all money out; write it as money left over.", ""),
        ("It averaged Rs 25,428.67 last month. Save more.", "It averaged Rs 25,428.67 last month. Save more."),
        ("Spent at SOME_SHOP and Rs 500 at Dr. Rao.", "Spent at SOME_SHOP and Rs 500 at Dr. Rao."),
    ],
)
def test_sentences_using_a_field_name_are_removed(text, kept):
    assert insights._clean_text(text) == kept


def test_the_instructions_state_how_long_the_emergency_target_is():
    assert f"{insights.EMERGENCY_FUND_MONTHS} months of average spending" in insights.SYSTEM


def test_the_instructions_do_not_say_if_for_what_python_decides():
    assert "latest_month_complete" not in insights.SYSTEM
    assert "below zero" not in insights.SYSTEM


def test_masking_does_not_hide_any_amount(history, fake_llm):
    """Whole amounts of six or more digits would be masked like account numbers if sent as ints."""
    history.post("/insights")
    assert "[NUM]" not in fake_llm.calls[-1]
    assert mask(fake_llm.calls[-1]) == fake_llm.calls[-1]


def test_report_is_saved_so_a_reload_uses_no_request(history, fake_llm):
    assert history.get("/insights").status_code == 404
    created = history.post("/insights").json()
    requests_used = len(fake_llm.calls)
    assert history.get("/insights").json() == created
    assert len(fake_llm.calls) == requests_used


def test_same_data_uses_the_saved_llm_answer(history, fake_llm):
    history.post("/insights")
    requests_used = len(fake_llm.calls)
    history.post("/insights")
    assert len(fake_llm.calls) == requests_used


def test_insights_still_work_when_the_market_is_down(client, fake_llm, market_down):
    add(client, "2026-10-10", 3000, "Food & Dining", "Swiggy")
    fake_llm.insights = {
        "summary": "Short.",
        "spending_cuts": [],
        "investment_advice": [{"product": "Gold", "percent_of_available": 0, "advice": "No live figures."}],
    }
    response = client.post("/insights")
    assert response.status_code == 200
    assert response.json()["investment_advice"][0]["fund"] is None
    assert sent_facts(fake_llm)["market"] == []


def test_llm_failure_is_reported_and_saves_nothing(history, fake_llm):
    fake_llm.fail = True
    response = history.post("/insights")
    assert response.status_code == 502
    assert "Rate limit" in response.json()["detail"]
    assert history.get("/insights").status_code == 404


def test_no_transactions_is_a_clear_error(client, fake_llm, live_market):
    response = client.post("/insights")
    assert response.status_code == 400
    assert fake_llm.calls == []
