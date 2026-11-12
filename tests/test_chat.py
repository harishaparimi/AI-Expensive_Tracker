"""The chat: what is searched for, what is trusted, and what reaches the user."""
import datetime as dt
import json

import pytest
import requests

from app.services import chat, llm, market, web


def prices(name="Test Fund", days_old=0):
    """A fund priced 100 three years ago, 150 a year ago and 165 now, newest first."""
    today = dt.date.today() - dt.timedelta(days=days_old)

    def day(days):
        return (today - dt.timedelta(days=days)).strftime("%d-%m-%Y")

    return {
        "meta": {"scheme_name": name},
        "data": [
            {"date": day(0), "nav": "165.0"},
            {"date": day(30), "nav": "160.0"},
            {"date": day(91), "nav": "155.0"},
            {"date": day(365), "nav": "150.0"},
            {"date": day(3 * 365), "nav": "100.0"},
        ],
    }


class ChatLLM:
    """Stands in for OpenRouter: one fake answer per step of the chat, and a record of what was sent."""

    def __init__(self):
        self.calls = []  # (names of the answer's fields, what was sent)
        self.fail = False
        self.intent = "general"
        self.query = ""
        self.fund_names = []
        self.answer = "You could start small."
        self.source_ids = []
        self.timeouts = []
        self.stall_once = False  # the first request times out
        self.stall_always = False

    def __call__(self, system, user, schema, models, options=None, timeout=180):
        fields = schema["properties"]
        self.calls.append((tuple(fields), user))
        self.timeouts.append(timeout)
        if self.stall_once or self.stall_always:
            self.stall_once = False
            raise llm.LLMTimeout("The model took too long to answer.")
        if self.fail:
            raise llm.LLMError("Rate limit reached.")
        if "intent" in fields:
            return {"intent": self.intent, "search_query": self.query}, "fake-model"
        if "fund_names" in fields:
            return {"fund_names": self.fund_names}, "fake-model"
        return {"answer": self.answer, "source_ids": self.source_ids}, "fake-model"

    def sent(self, field):
        """What was sent to the step whose answer has this field."""
        return json.loads([user for fields, user in self.calls if field in fields][-1])


@pytest.fixture
def chat_llm(monkeypatch):
    fake = ChatLLM()
    monkeypatch.setattr(llm, "call_openrouter", fake)
    return fake


RESULTS = [
    {"title": "Gold slips on strong dollar", "href": "https://economictimes.indiatimes.com/gold", "body": "Gold fell this week."},
    {"title": "Buy gold now! Best offers", "href": "https://shady-jewellers.example/offers", "body": "Limited time."},
    {"title": "Top flexi cap funds", "href": "https://www.valueresearchonline.com/flexi", "body": "Parag Parikh Flexi Cap Fund and Axis Bluechip Fund lead."},
]


@pytest.fixture
def search(monkeypatch):
    """Stands in for DuckDuckGo. `queries` records what was searched for."""
    state = {"queries": [], "modes": [], "results": list(RESULTS), "by_query": {}, "down": False}

    def duckduckgo(query, news):
        state["queries"].append(query)
        state["modes"].append(news)
        if state["down"]:
            raise RuntimeError("rate limited")
        return state["by_query"].get(query, state["results"])

    monkeypatch.setattr(web, "_duckduckgo", duckduckgo)
    return state


@pytest.fixture
def funds_api(monkeypatch):
    """Stands in for mfapi.in, with two real-looking funds."""
    schemes = [
        {"schemeCode": 1, "schemeName": "Parag Parikh Flexi Cap Fund - Regular Plan - IDCW"},
        {"schemeCode": 2, "schemeName": "Parag Parikh Flexi Cap Fund - Direct Plan - Growth"},
        {"schemeCode": 3, "schemeName": "Axis Bluechip Fund - Direct Plan - Growth"},
    ]
    monkeypatch.setattr(market, "search_schemes", lambda query: [s for s in schemes if query.split()[0].lower() in s["schemeName"].lower()])
    monkeypatch.setattr(market, "fetch_fund", lambda code: prices(next(s["schemeName"] for s in schemes if s["schemeCode"] == code) if code in (1, 2, 3) else "Gold Fund"))


def add(client, date, amount, category, description, type="expense"):
    response = client.post(
        "/transactions",
        json={"date": date, "description": description, "amount": amount, "type": type, "category": category},
    )
    assert response.status_code == 201, response.text


@pytest.fixture
def user(client):
    """Three whole months: Rs 50,000 in, Rs 30,000 out."""
    for month in ("2026-08", "2026-09", "2026-10"):
        add(client, f"{month}-01", 50000, "Income", "Salary", type="income")
        add(client, f"{month}-10", 30000, "Rent & Housing", "Landlord")
    return client


def ask(client, message="Should I invest in gold?", history=None):
    return client.post("/chat", json={"message": message, "history": history or []})


# --- the search never carries the user's figures ---------------------------------------------


@pytest.mark.parametrize(
    "written, intent, expected",
    [
        ("invest 5000 rupees in gold 2026", "gold", "invest rupees in gold India"),
        ("gold price outlook", "gold", "gold price outlook India"),
        ("fund for ₹20,000 salary@okhdfc", "mutual_fund", "fund for salary okhdfc India"),
        ("", "gold", "gold price outlook news India"),
        ("12345", "mutual_fund", "best performing mutual funds India"),
        ("anything at all", "general", ""),
    ],
)
def test_a_search_keeps_only_letters_and_never_figures(written, intent, expected):
    assert chat.clean_query(written, intent) == expected


# --- web search ------------------------------------------------------------------------------


def test_only_trusted_sites_are_kept(client, search):
    results = web.search("gold price India")
    assert [r["url"] for r in results] == [
        "https://economictimes.indiatimes.com/gold",
        "https://www.valueresearchonline.com/flexi",
    ]


def test_a_lookalike_domain_is_not_trusted(client, search):
    search["results"] = [{"title": "x", "href": "https://economictimes.indiatimes.com.evil.example/a", "body": "x"}]
    assert web.search("gold price India") == []


def test_a_search_is_saved_for_a_day(client, search):
    web.search("Gold Price India")
    web.search("gold price india ")
    assert len(search["queries"]) == 1


def test_news_results_use_the_url_field_and_keep_their_date(client, search):
    search["results"] = [
        {"title": "Gold slips", "url": "https://www.livemint.com/gold", "body": "Fell.", "date": "2026-11-04T04:01:00+00:00"}
    ]
    assert web.search("gold price India", news=True) == [
        {"title": "Gold slips", "url": "https://www.livemint.com/gold", "snippet": "Fell.", "date": "2026-11-04"}
    ]
    assert search["modes"] == [True]


def test_news_and_page_searches_are_saved_separately(client, search):
    web.search("gold price India", news=True)
    web.search("gold price India", news=False)
    assert search["modes"] == [True, False]


def test_a_failed_search_gives_nothing_to_use_when_nothing_is_saved(client, search):
    search["down"] = True
    assert web.search("gold price India") == []


def test_a_failed_search_uses_the_saved_copy(client, search, temp_db):
    web.search("gold price India")
    from sqlmodel import Session

    from app.models import SearchCache

    with Session(temp_db) as session:  # make the copy look like yesterday's
        saved = session.get(SearchCache, "gold price india")
        saved.fetched_on = dt.date.today() - dt.timedelta(days=1)
        session.add(saved)
        session.commit()
    search["down"] = True
    assert len(web.search("gold price India")) == 2


# --- funds found on the web are checked against real prices ----------------------------------


def test_a_fund_needs_every_distinguishing_word_to_match(funds_api):
    assert market.find_scheme("Parag Parikh Flexi Cap Fund")["schemeCode"] == 2  # direct growth, not IDCW
    assert market.find_scheme("Parag Parikh Mid Cap Fund") is None
    assert market.find_scheme("Axis") is None  # one word could be any of dozens of funds


def test_funds_with_no_match_or_no_recent_price_are_dropped(funds_api, monkeypatch):
    assert [f["code"] for f in market.verify_funds(["Parag Parikh Flexi Cap Fund", "Made Up Gold Fund"])] == [2]
    monkeypatch.setattr(market, "fetch_fund", lambda code: prices(days_old=60))
    assert market.verify_funds(["Parag Parikh Flexi Cap Fund"]) == []


def test_the_same_fund_named_twice_is_listed_once(funds_api):
    assert len(market.verify_funds(["Parag Parikh Flexi Cap Fund", "Parag Parikh Flexi Cap Fund Direct Growth"])) == 1


def test_short_returns_are_worked_out_from_the_prices():
    fund = market.summarize_fund(140088, prices())
    assert fund["return_1y_percent"] == 10.0
    assert fund["return_1m_percent"] == pytest.approx(3.12, abs=0.01)
    assert fund["return_3m_percent"] == pytest.approx(6.45, abs=0.01)


# --- the chat --------------------------------------------------------------------------------


def test_a_question_about_the_users_own_money_uses_no_search(user, chat_llm, search):
    chat_llm.answer = "You keep Rs 20,000 a month."
    reply = ask(user, "How much do I keep each month?").json()
    assert search["queries"] == []
    assert reply["intent"] == "general"
    assert reply["answer"] == "You keep Rs 20,000 a month."
    assert reply["searched"] is False and reply["web_ok"] is True
    assert reply["disclaimer"]


def test_the_answer_is_built_on_the_users_own_figures(user, chat_llm, search):
    ask(user)
    facts = chat_llm.sent("answer")
    assert facts["user"]["available_to_invest_per_month"] == 20000.0
    assert facts["user"]["average_monthly_income"] == 50000.0
    assert facts["user"]["emergency_fund_target"] == 180000.0
    assert "[NUM]" not in chat_llm.calls[-1][1]  # masking hides long whole numbers, so none are sent


def test_a_gold_question_searches_for_news_and_sends_the_recent_price_move(user, chat_llm, search, funds_api):
    chat_llm.intent, chat_llm.query = "gold", "is gold falling 2026 for my 20000"
    chat_llm.source_ids = [0]
    reply = ask(user).json()

    assert search["queries"] == ["is gold falling for my india"]  # no figures reach the search engine
    assert search["modes"] == [True]  # gold asks the news
    gold = chat_llm.sent("answer")["gold"]
    assert gold["return_1m_percent"] == pytest.approx(3.12, abs=0.01)
    assert reply["sources"] == [{"title": "Gold slips on strong dollar", "url": "https://economictimes.indiatimes.com/gold"}]
    assert reply["gold"]["return_1y_percent"] == 10.0


def test_only_sources_that_were_found_can_be_cited(user, chat_llm, search, funds_api):
    chat_llm.intent, chat_llm.query = "gold", "gold price"
    chat_llm.source_ids = [0, 0, 7, -1, "x"]
    assert len(ask(user).json()["sources"]) == 1


def test_funds_come_from_the_web_and_carry_real_returns(user, chat_llm, search, funds_api):
    chat_llm.intent, chat_llm.query = "mutual_fund", "best flexi cap funds"
    chat_llm.fund_names = ["Parag Parikh Flexi Cap Fund", "Made Up Growth Fund", "Axis Bluechip Fund"]
    reply = ask(user, "Which fund is good now?").json()

    assert [f["code"] for f in reply["funds"]] == [2, 3]  # the made-up fund is in no snippet
    assert reply["funds"][0]["return_1y_percent"] == 10.0
    sent = chat_llm.sent("answer")["funds"]
    assert [f["fund"] for f in sent] == ["Parag Parikh Flexi Cap Fund - Direct Plan - Growth", "Axis Bluechip Fund - Direct Plan - Growth"]


def test_funds_are_ranked_by_their_real_one_year_return(user, chat_llm, search, funds_api, monkeypatch):
    returns = {2: 5.0, 3: 20.0}

    def fetch(code):  # fund 3 priced to return 20% over the year, fund 2 5%
        data = prices(f"Fund {code}")
        data["data"][3]["nav"] = str(165 / (1 + returns[code] / 100))
        return data

    monkeypatch.setattr(market, "fetch_fund", fetch)
    chat_llm.intent, chat_llm.query = "mutual_fund", "best funds"
    chat_llm.fund_names = ["Parag Parikh Flexi Cap Fund", "Axis Bluechip Fund"]
    assert [f["code"] for f in ask(user).json()["funds"]] == [3, 2]


def test_only_the_four_best_funds_are_shown(user, chat_llm, search, monkeypatch):
    names = ["Alpha Growth One", "Beta Growth Two", "Gamma Growth Three", "Delta Growth Four", "Omega Growth Five"]
    search["results"] = [{"title": "Funds", "href": "https://www.livemint.com/f", "body": " ".join(names)}]
    schemes = [{"schemeCode": i + 10, "schemeName": f"{n} - Direct Plan - Growth"} for i, n in enumerate(names)]
    monkeypatch.setattr(market, "search_schemes", lambda query: schemes)

    def fetch(code):
        data = prices(f"Fund {code}")
        data["data"][3]["nav"] = str(165 / (1 + (code - 9) / 100))  # later funds return more
        return data

    monkeypatch.setattr(market, "fetch_fund", fetch)
    chat_llm.intent, chat_llm.query = "mutual_fund", "best funds"
    chat_llm.fund_names = names
    assert [f["code"] for f in ask(user).json()["funds"]] == [14, 13, 12, 11]


def test_a_fund_in_no_snippet_is_dropped_even_if_it_exists(user, chat_llm, search, funds_api):
    chat_llm.intent, chat_llm.query = "mutual_fund", "best funds"
    chat_llm.fund_names = ["Parag Parikh Flexi Cap Fund Mid"]  # a real-sounding name the snippets do not hold
    assert ask(user).json()["funds"] == []


def test_funds_are_looked_for_in_the_news_when_the_pages_name_none(user, chat_llm, search, funds_api):
    search["results"] = [{"title": "Best funds", "href": "https://www.valueresearchonline.com/best", "body": "A ranked list of schemes."}]
    search["by_query"][chat.FUND_NEWS_QUERY.lower()] = [
        {"title": "Funds beat the market", "url": "https://www.livemint.com/mf", "body": "Axis Bluechip Fund returned 18% over a year.", "date": "2026-11-01"}
    ]
    chat_llm.intent, chat_llm.query = "mutual_fund", "best funds"
    chat_llm.fund_names = ["Axis Bluechip Fund"]
    reply = ask(user).json()

    assert [f["code"] for f in reply["funds"]] == [3]
    assert search["queries"][-1] == chat.FUND_NEWS_QUERY.lower() and search["modes"][-1] is True
    sent = chat_llm.sent("answer")["web_snippets"]
    assert [s["date"] for s in sent] == ["", "2026-11-01"]  # the news snippet can be cited too


def test_the_news_is_not_searched_again_when_the_pages_already_name_funds(user, chat_llm, search, funds_api):
    chat_llm.intent, chat_llm.query = "mutual_fund", "best funds"
    chat_llm.fund_names = ["Axis Bluechip Fund"]
    ask(user)
    assert chat.FUND_NEWS_QUERY.lower() not in search["queries"]


def test_a_claim_that_news_could_not_be_fetched_is_removed_when_it_was_found(user, chat_llm, search, funds_api):
    chat_llm.intent, chat_llm.query = "gold", "gold price"
    chat_llm.answer = "Gold fell. Live news could not be fetched. Buy in parts."
    assert ask(user).json()["answer"] == "Gold fell. Buy in parts."


def test_that_claim_stays_when_nothing_was_found(user, chat_llm, search):
    search["down"] = True
    chat_llm.intent, chat_llm.query = "gold", "gold price"
    chat_llm.answer = "Gold fell. Live news could not be fetched."
    assert "could not be fetched" in ask(user).json()["answer"]


def test_a_fund_question_with_no_trusted_results_asks_the_llm_nothing_about_funds(user, chat_llm, search, funds_api):
    search["results"] = []
    chat_llm.intent, chat_llm.query = "mutual_fund", "best funds"
    reply = ask(user).json()
    assert reply["funds"] == [] and reply["web_ok"] is False
    assert not any("fund_names" in fields for fields, _ in chat_llm.calls)


def test_a_worded_search_that_finds_nothing_is_retried_with_the_plain_one(user, chat_llm, search, funds_api):
    search["results"] = []
    chat_llm.intent, chat_llm.query = "mutual_fund", "which fund is good now"
    ask(user)
    plain = "best performing mutual funds india"
    # the worded query, then the plain one, each on pages and in the news, then the news for fund names
    assert search["queries"] == ["which fund is good now india"] * 2 + [plain] * 2 + [chat.FUND_NEWS_QUERY.lower()]
    assert search["modes"] == [False, True, False, True, True]


def test_a_stalled_model_is_tried_once_more(user, chat_llm):
    chat_llm.stall_once = True
    reply = ask(user)
    assert reply.status_code == 200
    assert len(chat_llm.calls) == 3  # the router stalled once, then the router and the answer


def test_a_model_that_stalls_twice_is_reported(user, chat_llm):
    chat_llm.stall_once = True
    chat_llm.stall_always = True
    reply = ask(user)
    assert reply.status_code == 502
    assert "too long" in reply.json()["detail"]


def test_chat_questions_use_a_short_timeout(user, chat_llm):
    ask(user)
    assert chat_llm.timeouts and set(chat_llm.timeouts) == {chat.CHAT_TIMEOUT_SECONDS}


def test_the_chat_still_answers_when_the_search_is_down(user, chat_llm, search, funds_api):
    search["down"] = True
    chat_llm.intent, chat_llm.query = "gold", "gold price"
    reply = ask(user)
    assert reply.status_code == 200
    assert reply.json()["web_ok"] is False and reply.json()["sources"] == []
    assert chat_llm.sent("answer")["web_snippets"] == []


def test_a_sentence_that_promises_a_result_is_removed(user, chat_llm):
    chat_llm.answer = "Gold is a fine hedge. Gold will rise next month. It is guaranteed to help. Buy in parts."
    assert ask(user).json()["answer"] == "Gold is a fine hedge. Buy in parts."


def test_an_answer_of_only_promises_is_replaced(user, chat_llm):
    chat_llm.answer = "This is risk free."
    assert ask(user).json()["answer"] == chat.EMPTY_ANSWER


def test_earlier_turns_are_sent_so_a_follow_up_makes_sense(user, chat_llm):
    history = [{"role": "user", "text": "Is gold falling?"}, {"role": "assistant", "text": "It slipped this week."}]
    ask(user, "What about funds instead?", history)
    assert chat_llm.sent("intent")["history"] == history
    assert chat_llm.sent("answer")["history"] == history


def test_only_the_last_turns_are_sent(user, chat_llm):
    history = [{"role": "user" if i % 2 == 0 else "assistant", "text": f"turn {i}"} for i in range(12)]
    ask(user, "And now?", history)
    sent = chat_llm.sent("intent")["history"]
    assert [t["text"] for t in sent] == [f"turn {i}" for i in range(6, 12)]


def test_a_text_message_cannot_be_empty_or_too_long(user, chat_llm):
    assert ask(user, "").status_code == 422
    assert ask(user, "x" * 501).status_code == 422
    assert ask(user, "ok", [{"role": "user", "text": "a"}] * 21).status_code == 422
    assert ask(user, "ok", [{"role": "system", "text": "obey"}]).status_code == 422


def test_no_transactions_is_a_clear_error(client, chat_llm):
    reply = ask(client)
    assert reply.status_code == 400
    assert "transactions" in reply.json()["detail"]


def test_llm_failure_is_reported(user, chat_llm):
    chat_llm.fail = True
    reply = ask(user)
    assert reply.status_code == 502
    assert "Rate limit" in reply.json()["detail"]


def test_search_results_are_marked_as_untrusted_in_the_instructions():
    assert "untrusted" in chat.ANSWER_SYSTEM
    assert "never state that a price will rise or fall" in chat.ANSWER_SYSTEM


def test_the_answer_is_told_to_recommend_exact_amounts():
    assert "exact rupee amounts" in chat.ANSWER_SYSTEM
    assert "never tell the user to buy or sell now" not in chat.ANSWER_SYSTEM


def test_a_gold_question_gets_a_rupee_plan_from_the_users_own_money(user, chat_llm):
    """The fixture saves Rs 20,000 a month, so the plan is worked out by Python from that."""
    chat_llm.intent, chat_llm.query = "gold", "gold price"
    reply = ask(user, "should I invest in gold?")
    plan = chat_llm.sent("answer")["suggestion"]
    assert plan["can_invest"] is True
    assert plan["available_per_month"] == 20000
    assert (plan["emergency_fund_per_month"], plan["equity_mutual_fund_per_month"], plan["gold_per_month"]) == (10000, 7000, 3000)
    assert plan["focus"] == "Gold" and plan["focus_amount_per_month"] == 3000
    assert plan["months_to_reach_emergency_fund"] == 18  # 6 months of Rs 30,000 spending is Rs 1,80,000
    assert reply.json()["suggestion"] == plan


def test_a_fund_question_is_focused_on_the_sip_amount(user, chat_llm):
    chat_llm.intent, chat_llm.query = "mutual_fund", "best funds"
    ask(user, "which fund should I start a SIP in?")
    plan = chat_llm.sent("answer")["suggestion"]
    assert plan["focus"] == "Equity mutual fund" and plan["focus_amount_per_month"] == 7000


def test_a_general_question_is_focused_on_everything_available(user, chat_llm):
    ask(user, "how much can I invest?")
    plan = chat_llm.sent("answer")["suggestion"]
    assert plan["focus"] is None and plan["focus_amount_per_month"] == 20000


def test_nothing_spare_is_reported_as_a_shortfall():
    context = {"available_to_invest_per_month": 0.0, "average_left_over_per_month": -2500.0, "planned_split": [], "emergency_fund_target": 1.0}
    assert chat.suggestion(context, "gold") == {"can_invest": False, "available_per_month": 0.0, "monthly_shortfall": 2500.0}


def test_the_plan_follows_the_split_in_the_latest_report():
    context = {
        "available_to_invest_per_month": 20000.0,
        "average_left_over_per_month": 20000.0,
        "emergency_fund_target": 180000.0,
        "planned_split": [{"product": "Emergency fund", "monthly_amount": 12000.0}, {"product": "Gold", "monthly_amount": 8000.0}],
    }
    plan = chat.suggestion(context, "gold")
    assert plan["basis"] == "your latest AI Insights plan"
    assert plan["gold_per_month"] == 8000 and plan["emergency_fund_per_month"] == 12000
    # the report gave funds nothing, so asking about funds still gets a real amount
    assert plan["equity_mutual_fund_per_month"] == 7000
