"""Web search for the chat: DuckDuckGo, trusted sites only, short snippets, saved for a day.

Only titles and snippets are used, never whole pages, so little untrusted text reaches the LLM.
The query is always generic. Nothing about the user is ever sent to the search engine.
"""
import datetime as dt
import json
from urllib.parse import urlparse

from sqlmodel import Session

from app import db
from app.models import SearchCache

# Finance results are full of pages written to sell a product, so only these sites are kept.
TRUSTED_DOMAINS = (
    "economictimes.indiatimes.com",
    "moneycontrol.com",
    "livemint.com",
    "business-standard.com",
    "financialexpress.com",
    "thehindubusinessline.com",
    "thehindu.com",
    "hindustantimes.com",
    "indianexpress.com",
    "timesofindia.indiatimes.com",
    "news18.com",
    "cnbctv18.com",
    "fortuneindia.com",
    "ndtvprofit.com",
    "reuters.com",
    "goodreturns.in",
    "valueresearchonline.com",
    "morningstar.in",
    "indmoney.com",
    "amfiindia.com",
    "sebi.gov.in",
    "rbi.org.in",
    "incometax.gov.in",
    "gold.org",
)
RESULTS_FETCHED = 15  # most results come from other sites, so ask for many and keep the trusted ones
RESULTS_KEPT = 5
TITLE_CHARS = 150
SNIPPET_CHARS = 350


def _trusted(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return any(host == domain or host.endswith("." + domain) for domain in TRUSTED_DOMAINS)


def _duckduckgo(query: str, news: bool) -> list[dict]:
    """The one place that talks to DuckDuckGo, so tests can replace it. News search returns dated
    articles, which suits "is gold falling"; plain search returns pages, which suits fund lists."""
    from ddgs import DDGS

    search_engine = DDGS()
    method = search_engine.news if news else search_engine.text
    return method(query, region="in-en", max_results=RESULTS_FETCHED)


def search(query: str, news: bool = False) -> list[dict]:
    """Trusted results as {title, url, snippet, date}. An empty list means nothing could be fetched,
    and the caller carries on without it. Yesterday's saved copy is used if the search fails."""
    key = ("news: " if news else "") + query.strip().lower()
    today = dt.date.today()
    with Session(db.shared_engine) as session:
        saved = session.get(SearchCache, key)
        if saved and saved.fetched_on == today:
            return json.loads(saved.results)

        try:
            found = _duckduckgo(query.strip().lower(), news)
        except Exception:  # the search library raises many types: rate limits, network, parsing
            return json.loads(saved.results) if saved else []

        # Text results call the link "href" and news results call it "url"
        links = [(item, item.get("href") or item.get("url") or "") for item in found]
        results = [
            {
                "title": str(item.get("title", ""))[:TITLE_CHARS],
                "url": url,
                "snippet": str(item.get("body", ""))[:SNIPPET_CHARS],
                "date": str(item.get("date", ""))[:10],
            }
            for item, url in links
            if url.startswith(("http://", "https://")) and _trusted(url)
        ][:RESULTS_KEPT]
        if results:  # an empty answer is not saved, so the next question tries again
            session.merge(SearchCache(query=key, fetched_on=today, results=json.dumps(results)))
            session.commit()
        return results
