"""Live fund prices from mfapi.in. Returns are worked out here, not by the LLM."""
import datetime as dt
import json
import re
from concurrent.futures import ThreadPoolExecutor

import requests
from sqlmodel import Session

from app import db
from app.models import FundSnapshot

MFAPI_URL = "https://api.mfapi.in/mf"
TIMEOUT_SECONDS = 20

# Examples of each kind of product, not recommendations of these particular funds.
FUNDS = {
    118482: "Equity mutual fund",
    140088: "Gold",
    143269: "Emergency fund",
}

# Words every fund name has, so they say nothing about which fund it is.
GENERIC_WORDS = {"fund", "plan", "growth", "direct", "regular", "scheme", "option", "mutual", "the", "of", "and"}
MIN_NAME_WORDS = 2  # "Axis" alone could be any of dozens of funds
MAX_NAV_AGE_DAYS = 14  # a fund with no recent price has been closed or merged
FUND_LOOKUP_WORKERS = 4


def fetch_fund(code: int) -> dict:
    """Get a fund's name and full price history, newest first."""
    response = requests.get(f"{MFAPI_URL}/{code}", timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def search_schemes(query: str) -> list[dict]:
    """Funds whose name matches, as {schemeCode, schemeName}."""
    response = requests.get(f"{MFAPI_URL}/search", params={"q": query}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _parse_date(text: str) -> dt.date:
    return dt.datetime.strptime(text, "%d-%m-%Y").date()


def _nav_on_or_before(history: list[dict], target: dt.date) -> float | None:
    """The price on a date, or the last one before it for weekends and holidays."""
    for row in history:
        if _parse_date(row["date"]) <= target:
            return float(row["nav"])
    return None  # the fund is younger than that


def _yearly_return(nav: float, past_nav: float | None, years: int) -> float | None:
    """Percent per year (CAGR). Over one year this is the plain return."""
    if past_nav is None:
        return None
    return round(((nav / past_nav) ** (1 / years) - 1) * 100, 2)


def _period_return(nav: float, past_nav: float | None) -> float | None:
    """Plain percent change since a past price, for short periods."""
    if past_nav is None:
        return None
    return round((nav / past_nav - 1) * 100, 2)


def summarize_fund(code: int, data: dict) -> dict:
    """1Y and 3Y are the long figures: the gold fund's 2019 unit split would distort longer periods.
    1M and 3M show the recent move, which is what a question like "is gold falling" needs."""
    history = data["data"]
    nav = float(history[0]["nav"])
    nav_date = _parse_date(history[0]["date"])

    def past(days: int) -> float | None:
        return _nav_on_or_before(history, nav_date - dt.timedelta(days=days))

    return {
        "code": code,
        "product": FUNDS.get(code, "Fund"),
        "fund": data["meta"]["scheme_name"],
        "nav": nav,
        "nav_date": nav_date.isoformat(),
        "return_1m_percent": _period_return(nav, past(30)),
        "return_3m_percent": _period_return(nav, past(91)),
        "return_1y_percent": _yearly_return(nav, past(365), 1),
        "return_3y_percent_per_year": _yearly_return(nav, past(3 * 365), 3),
    }


def _cached_fund(session: Session, code: int, today: dt.date) -> dict | None:
    """One fund's figures, fetched at most once a day, with the last saved copy if the API fails."""
    saved = session.get(FundSnapshot, code)
    if saved and saved.fetched_on == today:
        return {**json.loads(saved.snapshot), "stale": False}
    try:
        snapshot = summarize_fund(code, fetch_fund(code))
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        if saved:  # with no saved copy either, leave the fund out rather than make up numbers
            return {**json.loads(saved.snapshot), "stale": True}
        return None
    session.merge(FundSnapshot(code=code, fetched_on=today, snapshot=json.dumps(snapshot)))
    session.commit()  # now, so the next slow fetch does not block other browsers
    return {**snapshot, "stale": False}


def get_funds() -> list[dict]:
    """Today's figures for the example funds."""
    today = dt.date.today()
    with Session(db.shared_engine) as session:
        funds = (_cached_fund(session, code, today) for code in FUNDS)
        return [fund for fund in funds if fund]


def get_fund(code: int) -> dict | None:
    """Today's figures for one example fund, or None if it cannot be fetched."""
    with Session(db.shared_engine) as session:
        return _cached_fund(session, code, dt.date.today())


def name_words(name: str) -> list[str]:
    """The words of a fund name that tell it apart from other funds."""
    return [word for word in re.findall(r"[a-z0-9]+", name.lower()) if word not in GENERIC_WORDS]


def _scheme_score(scheme_name: str) -> int:
    """Prefer the direct growth plan, which is the usual one to compare, over dividend payouts."""
    name = scheme_name.lower()
    return 2 * ("direct" in name) + 2 * ("growth" in name) - 5 * ("idcw" in name or "dividend" in name)


def find_scheme(name: str) -> dict | None:
    """The mfapi.in fund whose name has every distinguishing word of `name`, or None. A name that
    matches nothing is dropped, so a fund the LLM made up can never reach the user."""
    words = name_words(name)
    if len(words) < MIN_NAME_WORDS:
        return None
    for count in dict.fromkeys((min(len(words), 4), min(len(words), 3), MIN_NAME_WORDS)):
        results = search_schemes(" ".join(words[:count]))
        matches = [s for s in results if all(word in s["schemeName"].lower() for word in words)]
        if matches:
            return max(matches, key=lambda s: _scheme_score(s["schemeName"]))
    return None


def verify_funds(names: list[str]) -> list[dict]:
    """Look each name up on mfapi.in and work out its returns. Names that cannot be matched to a
    fund with a recent price are left out."""

    def verify(name: str) -> dict | None:
        try:
            scheme = find_scheme(name)
            if scheme is None:
                return None
            summary = summarize_fund(scheme["schemeCode"], fetch_fund(scheme["schemeCode"]))
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
            return None
        age = dt.date.today() - dt.date.fromisoformat(summary["nav_date"])
        return summary if age.days <= MAX_NAV_AGE_DAYS else None

    with ThreadPoolExecutor(max_workers=FUND_LOOKUP_WORKERS) as pool:
        found = list(pool.map(verify, names))

    funds, seen = [], set()
    for fund in found:
        if fund and fund["code"] not in seen:
            seen.add(fund["code"])
            funds.append({**fund, "stale": False})
    return funds
