"""How good are the AI insights, and what does each LLM setting change?

Run with:  python -m scripts.eval_insights --settings default low off --runs 3 --out answers.json
Re-score:  python -m scripts.eval_insights --rescore answers.json   (uses no requests)
Scenarios: --scenario partial_month or deficit, because the demo data alone has neither. Pass the
           same one with --rescore, so the answers are scored against the same facts.

Every run is one real request. It reads the transactions and budgets in samples/seed.db, which is
built from samples/demo_statement.csv. The planted patterns below only apply to that demo data.
"""
import calendar
import datetime as dt
import json
import re
import sys

from sqlmodel import Session, select

from app import db
from app.models import CATEGORIES, Budget
from app.services import analytics, insights, market
from app.services.llm import mask
from scripts.eval_common import parse_args, save_or_load

# What scripts/make_demo_data.py planted, and the merchants that show each one.
PLANTED = {
    "food delivery growing": ("swiggy", "zomato"),
    "coffee growing": ("third wave",),
    "shopping spike": ("myntra", "amazon"),
    "gym paid every month": ("cult",),
}
ECHOED_INSTRUCTION = re.compile(r"write it as|share to 0|0 percent|never suggest", re.IGNORECASE)
RUPEE_FIGURE = re.compile(r"(?:Rs\.?|₹|INR)\s*(\d[\d,]*(?:\.\d+)?)\s*(lakh|crore)?", re.IGNORECASE)
MULTIPLIER = {"lakh": 100_000, "crore": 10_000_000}
PARTIAL_MONTH_WORDS = re.compile(r"incomplete|partly|partial|not complete", re.IGNORECASE)
COVERAGE_WORDS = re.compile(r"covered|complete|partly|partial", re.IGNORECASE)
DEFICIT_WORDS = re.compile(
    r"below zero|left over[^.]*negative|fix\w* (?:your )?spending|spending (?:comes )?(?:first|before)"
    r"|overspend|more than (?:you earn|comes in)",
    re.IGNORECASE,
)
MONTH_NAMES = list(calendar.month_name)[1:]
MONTH_FIGURE = re.compile(rf"(?:Rs\.?|₹)\s*(\d[\d,]*(?:\.\d+)?) in ({'|'.join(MONTH_NAMES)})", re.IGNORECASE)
MONTHS_OF_COVER = re.compile(r"\b(\w+)[ -]months?\b(?= of)", re.IGNORECASE)
SPENT_TOO_MUCH = re.compile(
    r"exceed|pass\w* (?:your )?income|(?<!below )(?<!not go )\b(?:above|over) (?:your )?income|more than (?:you earn|your income)|overspen",
    re.IGNORECASE,
)


def rupee_figures(text: str) -> list[tuple[float, bool]]:
    """Every rupee amount in a text, with whether it was rounded to lakh or crore."""
    figures = []
    for number, unit in RUPEE_FIGURE.findall(text):
        value = float(number.replace(",", "")) * MULTIPLIER.get(unit.lower(), 1)
        figures.append((value, bool(unit)))
    return figures


def numbers_in(value) -> set[float]:
    """Every number anywhere in the facts, ignoring its sign."""
    if isinstance(value, dict):
        return set().union(*map(numbers_in, value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*map(numbers_in, value)) if value else set()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {abs(float(value))}
    return set()


def is_traceable(value: float, rounded: bool, known: set[float]) -> bool:
    """A figure must match to the rupee, or within 1% if it was written in lakh or crore."""
    return any(abs(value - number) <= (number * 0.01 if rounded else 1) for number in known)


def mislabelled_sentences(cut: dict, merchant_facts: dict) -> list[str]:
    """Sentences that give a merchant's monthly figure as if it were its category's, such as
    'Shopping rose from Rs 2,814', when Rs 2,814 was Amazon Pay's amount."""
    amounts = {float(v) for v in merchant_facts["by_month"].values()}
    name = cut["merchant"].lower().split()[0]
    found = []
    for sentence in re.split(r"(?<=[.;])\s+", cut["finding"]):
        lowered = sentence.lower()
        if name in lowered or not any(category.lower() in lowered for category in CATEGORIES):
            continue
        if any(is_traceable(value, False, amounts) for value, _ in rupee_figures(sentence)):
            found.append(sentence)
    return found


def quotes(text: str, value: float) -> bool:
    """Does the text state this rupee figure?"""
    return any(is_traceable(figure, rounded, {value}) for figure, rounded in rupee_figures(text))


def text_cut_off(raw_texts: list) -> tuple[int, int]:
    """How many fields lost a pasted copy of the answer, and how many lost some other extra text."""
    pasted = extra = 0
    for text in raw_texts:
        match = insights.PASTE_START.search(str(text or ""))
        if match:
            removed = str(text)[match.start():]
            is_copy = '{"' in removed or "merchant" in removed or "spending_cuts" in removed
            pasted += is_copy
            extra += not is_copy
    return pasted, extra


def false_budget_claims(cuts: list[dict], budgets: list[dict]) -> list[str]:
    """Findings saying a category is over budget when the facts say it is not."""
    claims = []
    for cut in cuts:
        finding = cut["finding"].lower()
        for budget in budgets:
            if budget["category"].lower() in finding and " over " in finding and budget["status"] != "over":
                claims.append(f"{cut['merchant']}: {budget['category']} is {budget['status']}")
    return claims


def wrong_month_figures(cut: dict, merchant_facts: dict) -> list[str]:
    """Pairs like 'Rs 2,262 in May' that do not match what this merchant spent in that month."""
    by_month = {month[5:]: amount for month, amount in merchant_facts["by_month"].items()}
    wrong = []
    for figure, month_name in MONTH_FIGURE.findall(cut["finding"]):
        month = f"{MONTH_NAMES.index(month_name.capitalize()) + 1:02d}"
        if abs(float(figure.replace(",", "")) - by_month.get(month, 0)) > 1:
            wrong.append(f"{cut['merchant']}: Rs {figure} in {month_name}, actual {by_month.get(month, 0)}")
    return wrong


def wrong_months_of_cover(emergency_advice: str) -> list[str]:
    """Phrases like 'three months of spending', for a target that is six months."""
    return [
        word
        for word in MONTHS_OF_COVER.findall(emergency_advice)
        if word.lower() not in ("six", str(insights.EMERGENCY_FUND_MONTHS))
    ]


def deficit_claim_wrong(advice: list[str], average_left_over: float) -> bool:
    """The advice must say spending comes first if money ran short on average, and only then."""
    return bool(DEFICIT_WORDS.search(" ".join(advice))) != (average_left_over < 0)


def yearly_cost_stated(cuts: list[dict], recurring: list[dict]) -> bool:
    """Does a kept cut give the yearly cost of the top recurring payment? True if there is none."""
    if not recurring:
        return True
    top = recurring[0]
    return any(
        cut["merchant"] == top["merchant"] and quotes(f"{cut['finding']} {cut['action']}", top["yearly_cost"])
        for cut in cuts
    )


def false_overspending_claim(summary: str, totals: list[dict]) -> bool:
    """Saying spending exceeded income in a period when no month's spending did. Money left over can
    go negative because of money invested or transferred, without spending being above income."""
    return bool(SPENT_TOO_MUCH.search(summary)) and all(row["spending"] <= row["income"] for row in totals)


def partial_month(df):
    return df[df["date"] <= dt.date(2026, 10, 20)]


def deficit(df):
    halved = df.copy()
    halved.loc[halved["type"] == "income", "amount"] /= 2
    return halved


SCENARIOS = {"normal": lambda df: df, "partial_month": partial_month, "deficit": deficit}


def score(answer: dict, df, facts: dict, funds: list[dict]) -> dict:
    report = insights.check_answer(answer, df, facts, funds)
    cuts, advice = report["spending_cuts"], report["investment_advice"]
    raw = [answer["summary"]]
    raw += [text for cut in answer["spending_cuts"] for text in (cut.get("finding"), cut.get("action"))]
    raw += [item.get("advice") for item in answer["investment_advice"]]
    clean = [report["summary"]]
    clean += [text for cut in cuts for text in (cut["finding"], cut["action"])]
    clean += [item["advice"] for item in advice]

    known = numbers_in(facts)  # the LLM is not given the targets Python works out, so it cannot quote them
    figures = [figure for text in clean for figure in rupee_figures(text)]
    untraced = [value for value, rounded in figures if not is_traceable(value, rounded, known)]

    history = {m["merchant"]: m for m in facts["merchants"]}
    mislabelled = [
        sentence for cut in cuts if cut["merchant"] in history for sentence in mislabelled_sentences(cut, history[cut["merchant"]])
    ]
    wrong_months = [
        pair for cut in cuts if cut["merchant"] in history for pair in wrong_month_figures(cut, history[cut["merchant"]])
    ]
    merchants = [cut["merchant"] for cut in cuts]
    percents = [item["percent_of_available"] for item in advice]
    by_product = {item["product"]: item for item in advice}

    plan = facts["plan"]
    complete = insights.latest_month_complete(facts["data_through"])
    term_advice = by_product.get("Term insurance", {}).get("advice", "")
    pasted, extra_text = text_cut_off(raw)

    quoted, to_quote = 0, 0
    for fund in funds:
        if fund["product"] in ("Equity mutual fund", "Gold"):
            for key in ("return_1y_percent", "return_3y_percent_per_year"):
                if fund.get(key) is not None:
                    to_quote += 1
                    quoted += str(fund[key]) in by_product.get(fund["product"], {}).get("advice", "")

    return {
        "cuts_kept": len(cuts),
        "cuts_proposed": len(answer["spending_cuts"]),
        "rejected": report["rejected_merchants"],
        "figures": len(figures),
        "untraced": untraced,
        "pasted": pasted,
        "extra_text": extra_text,
        "emergency_target_quoted": quotes(by_product.get("Emergency fund", {}).get("advice", ""), plan["emergency_fund_target"]),
        "term_cover_quoted": quotes(term_advice, plan["term_cover_low"]) and quotes(term_advice, plan["term_cover_high"]),
        "field_names": sorted({name for text in clean for name in insights.FIELD_NAME.findall(text)}),
        "field_names_removed": sum(bool(insights.FIELD_NAME.search(str(text or ""))) for text in raw),
        "wrong_cover": wrong_months_of_cover(by_product.get("Emergency fund", {}).get("advice", "")),
        "echoed": [text[:90] for text in clean if ECHOED_INSTRUCTION.search(text)],
        "mislabelled": mislabelled,
        "wrong_months": wrong_months,
        "false_budget": false_budget_claims(cuts, facts["budgets_latest_month"]),
        # A complete month must not be called partial, and a partial one must be flagged.
        "month_claim_wrong": bool(PARTIAL_MONTH_WORDS.search(report["summary"])) == complete,
        "complete_month_mentioned": complete and bool(COVERAGE_WORDS.search(report["summary"])),
        "deficit_claim_wrong": deficit_claim_wrong([item["advice"] for item in advice], plan["average_left_over_last_3_months"]),
        "cut_count_ok": len(answer["spending_cuts"]) <= insights.MAX_CUTS,
        "yearly_cost_stated": yearly_cost_stated(cuts, facts["recurring_payments"]),
        "false_overspending": false_overspending_claim(report["summary"], facts["monthly_totals"]),
        "starts_with_merchant": sum(cut["finding"].lower().startswith(cut["merchant"].lower().split()[0]) for cut in cuts),
        "products_in_order": [item["product"] for item in advice] == insights.PRODUCTS,
        "percents_total_100": sum(percents) == 100,
        "emergency_largest": bool(percents) and by_product.get("Emergency fund", {}).get("percent_of_available", 0) == max(percents),
        "term_is_zero": by_product.get("Term insurance", {}).get("percent_of_available") == 0,
        "amounts_fit": sum(item["monthly_amount"] for item in advice) <= report["available_per_month"],
        "returns_quoted": quoted,
        "returns_to_quote": to_quote,
        "planted": [
            pattern
            for pattern, words in PLANTED.items()
            if any(word in merchant.lower() for merchant in merchants for word in words)
        ],
        "merchants": merchants,
        "total_saving": report["total_monthly_saving"],
        "available": report["available_per_month"],
    }


def print_setting(setting: str, results: list[dict], df, facts: dict, funds: list[dict]) -> None:
    done = [r for r in results if r["answer"]]
    print(f"\n=== {setting}: {len(done)} of {len(results)} runs answered")
    for result in results:
        if result["error"]:
            print(f"  failed: {result['error'][:200]}")
    if not done:
        return
    scores = [score(r["answer"], df, facts, funds) for r in done]
    runs = len(scores)

    def add(key):
        return sum(s[key] for s in scores)

    def add_lengths(key):
        return sum(len(s[key]) for s in scores)

    print(f"  seconds per run           {[r['seconds'] for r in done]}, model {done[0]['model']}")
    print(f"  cuts kept / proposed      {add('cuts_kept')} / {add('cuts_proposed')}, rejected {[m for s in scores for m in s['rejected']]}")
    print(f"  rupee figures traced      {add('figures') - add_lengths('untraced')} / {add('figures')}, misses {[v for s in scores for v in s['untraced']]}")
    print(f"  runs with a pasted copy   {sum(s['pasted'] > 0 for s in scores)} / {runs} before cleaning")
    print(f"  runs with other extra text {sum(s['extra_text'] > 0 for s in scores)} / {runs} before cleaning")
    print(f"  runs with field names left {sum(bool(s['field_names']) for s in scores)} / {runs} {sorted({n for s in scores for n in s['field_names']})}")
    print(f"  fields with names before cleaning {add('field_names_removed')}")
    print(f"  emergency target stated   {add('emergency_target_quoted')} / {runs}")
    print(f"  wrong months of cover     {[w for s in scores for w in s['wrong_cover']]}")
    print(f"  term cover range stated   {add('term_cover_quoted')} / {runs}")
    print(f"  instructions echoed       {add_lengths('echoed')} {[t for s in scores for t in s['echoed']]}")
    print(f"  mislabelled sentences     {add_lengths('mislabelled')}")
    for sentence in [t for s in scores for t in s["mislabelled"]]:
        print(f"      {sentence[:150]}")
    print(f"  figures against wrong month {[p for s in scores for p in s['wrong_months']]}")
    print(f"  false over-budget claims  {[p for s in scores for p in s['false_budget']]}")
    print(f"  wrong claim about the month {add('month_claim_wrong')} / {runs}")
    print(f"  complete month mentioned  {add('complete_month_mentioned')} / {runs}")
    print(f"  wrong claim about a deficit {add('deficit_claim_wrong')} / {runs}")
    print(f"  findings start w/ merchant {add('starts_with_merchant')} / {add('cuts_kept')}")
    print(f"  runs with at most {insights.MAX_CUTS} cuts   {add('cut_count_ok')} / {runs}")
    print(f"  recurring yearly cost stated {add('yearly_cost_stated')} / {runs}")
    print(f"  false overspending claims {add('false_overspending')} / {runs}")
    print(f"  products all there, in order {add('products_in_order')} / {runs}")
    print(f"  percents total 100        {add('percents_total_100')} / {runs}")
    print(f"  emergency fund largest    {add('emergency_largest')} / {runs}")
    print(f"  term insurance at 0%      {add('term_is_zero')} / {runs}")
    print(f"  amounts fit available     {add('amounts_fit')} / {runs}")
    print(f"  live returns quoted       {add('returns_quoted')} / {add('returns_to_quote')}")
    print(f"  distinct answers          {len({json.dumps(r['answer'], sort_keys=True) for r in done})} / {runs}")
    for pattern in PLANTED:
        print(f"  planted: {pattern:24s} {sum(pattern in s['planted'] for s in scores)} / {runs}")
    for s in scores:
        print(f"  run: saving {s['total_saving']}, available {s['available']}, cuts {s['merchants']}")


def main() -> None:
    args = parse_args("Score real insights answers.", ["default", "low", "off"], 3, list(SCENARIOS))
    db.init_db()
    with Session(db.sqlite_engine(db.SEED_PATH)) as session:
        df = SCENARIOS[args.scenario](analytics.load_transactions(session))
        if df.empty:
            sys.exit("No transactions found. Run python -m scripts.make_seed_db first.")
        funds = market.get_funds()
        facts = insights.build_facts(df, session.exec(select(Budget)).all(), funds)

    user = mask(json.dumps(facts, ensure_ascii=False))

    def make_jobs():
        return [
            {"setting": setting, "run": run, "system": insights.build_system(facts), "user": user, "schema": insights.SCHEMA}
            for setting in args.settings
            for run in range(args.runs)
        ]

    results = save_or_load(args, make_jobs)
    for setting in dict.fromkeys(r["setting"] for r in results):
        print_setting(setting, [r for r in results if r["setting"] == setting], df, facts, funds)


if __name__ == "__main__":
    main()
