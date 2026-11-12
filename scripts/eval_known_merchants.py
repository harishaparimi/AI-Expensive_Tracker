"""Does sending known merchant names keep spellings consistent, without causing wrong matches?

Run with:  python -m scripts.eval_known_merchants --runs 3 --out answers.json
Re-score:  python -m scripts.eval_known_merchants --rescore answers.json   (uses no requests)

Each run sends every case below twice, with the names and without them. The traps are as important
as the spellings, because a merged name puts a row in the wrong merchant's total. Everything here is
invented and hand labelled, so these are not measured rates on real statements.
"""
from app.services.categorizer import SCHEMA, SYSTEM, build_prompt
from app.services.llm import mask
from scripts.eval_common import parse_args, save_or_load

R = "612345678"  # the start of an invented reference number, which masking hides

KNOWN_NAMES = [
    "Swiggy",
    "Zomato",
    "Amazon Pay",
    "Uber",
    "Ola",
    "Netflix",
    "BookMyShow",
    "BigBasket",
    "Cult Fit",
    "Airtel",
    "BESCOM",
    "Apollo Pharmacy",
    "Groww",
    "Jio",
    "HDFC Bank",
    "ICICI Prudential MF",
    "Reliance Fresh",
    "HP Petrol Pump",
    "Flipkart",
]

# The same business spelled another way: narration, type, the known name it should reuse.
SAME = [
    ("POS 416021XXXXXX1234 BUNDL TECHNOLOGIES SWIGGY BANGALORE", "expense", "Swiggy"),
    (f"UPI/DR/{R}201/SWIGGY LIMITED/HDFC/swiggyupi@axb/Order", "expense", "Swiggy"),
    ("ZOMATO PVT LTD GURGAON POS DEBIT", "expense", "Zomato"),
    (f"UPI-ZOMATO ONLINE ORDER-zomato-order@ptybl-YESB0000123-{R}202-PAYMENT", "expense", "Zomato"),
    ("AMAZON PAY INDIA PRIVATE LIMITED", "expense", "Amazon Pay"),
    (f"UPI-AMAZONPAY-amazonpay@apl-UTIB0000444-{R}203-PAYMENT", "expense", "Amazon Pay"),
    ("POS 416021XXXXXX1234 UBER TRIP HELP.UBER.COM", "expense", "Uber"),
    (f"UPI-UBER INDIA SYSTEMS PVT LTD-uber@axisbank-UTIB0000222-{R}204-RIDE", "expense", "Uber"),
    (f"UPI/DR/{R}205/ANI TECHNOLOGIES OLA/YESB/olamoney@ybl/Ride", "expense", "Ola"),
    ("POS 416021XXXXXX1234 NETFLIX.COM MUMBAI", "expense", "Netflix"),
    (f"UPI-BOOK MY SHOW-bms@axisbank-UTIB0000777-{R}206-TICKETS", "expense", "BookMyShow"),
    (f"UPI-BIG BASKET-bb@axisbank-UTIB0000111-{R}207-PAYMENT", "expense", "BigBasket"),
    ("POS 416021XXXXXX1234 INNOVATIVE RETAIL BIGBASKET", "expense", "BigBasket"),
    (f"UPI-CULTFIT CUREFIT HEALTHCARE-cultfit@hdfcbank-HDFC0000044-{R}208-MEMBERSHIP", "expense", "Cult Fit"),
    (f"BIL/BPAY/{R}209/BHARTI AIRTEL LTD/9000000002", "expense", "Airtel"),
    (f"BILLPAY BANGALORE ELECTRICITY SUPPLY BESCOM-{R}210", "expense", "BESCOM"),
    (f"UPI-APOLLO PHARMACIES LTD-apollo@hdfcbank-HDFC0000222-{R}211-PAYMENT", "expense", "Apollo Pharmacy"),
    (f"ACH D- GROWW INVEST TECH PVT LTD SIP-{R}212", "expense", "Groww"),
    (f"BIL/BPAY/{R}213/RELIANCE JIO INFOCOMM/PREPAID", "expense", "Jio"),
    ("POS 416021XXXXXX1234 RELIANCE RETAIL FRESH HSR", "expense", "Reliance Fresh"),
    (f"UPI/DR/{R}214/FLIPKART INTERNET PVT LTD/AXIS/flipkart@axisbank/Order", "expense", "Flipkart"),
]

# A different business that shares a word: narration, type, the known name it must not take, categories.
TRAPS = [
    (f"UPI-SWIGGY INSTAMART-instamart@ybl-YESB0000123-{R}215-PAYMENT", "expense", "Swiggy", ["Groceries"]),
    (f"ACH D- AMAZON PRIME VIDEO MEMBERSHIP-{R}216", "expense", "Amazon Pay", ["Subscriptions"]),
    (f"UPI-AMAZON FRESH-amazonfresh@apl-UTIB0000444-{R}217-PAYMENT", "expense", "Amazon Pay", ["Groceries"]),
    (f"NEFT DR-UTIB0000123-OLA ELECTRIC MOBILITY-SCOOTER-{R}218", "expense", "Ola", ["Shopping", "Transport", "Other"]),
    (f"BILLPAY HP GAS CYLINDER REFILL-{R}219", "expense", "HP Petrol Pump", ["Bills & Utilities"]),
    ("POS 416021XXXXXX1234 RELIANCE DIGITAL FORUM MALL", "expense", "Reliance Fresh", ["Shopping"]),
    ("POS 416021XXXXXX1234 RELIANCE TRENDS HSR LAYOUT", "expense", "Reliance Fresh", ["Shopping"]),
    (f"UPI-JIOMART-jiomart@sbi-SBIN0000666-{R}220-ORDER", "expense", "Jio", ["Groceries", "Shopping"]),
    ("POS 416021XXXXXX1234 APOLLO TYRES SERVICE CENTRE", "expense", "Apollo Pharmacy", ["Transport", "Other", "Shopping"]),
    (f"ACH D- ICICI LOMBARD HEALTH INSURANCE-{R}221", "expense", "ICICI Prudential MF", ["Health", "Bills & Utilities", "Other"]),
    (f"ACH D- HDFC LIFE INSURANCE PREMIUM-{R}222", "expense", "HDFC Bank", ["Investments", "Bills & Utilities", "Other", "Health"]),
    ("POS 416021XXXXXX1234 BIG BAZAAR FUTURE RETAIL", "expense", "BigBasket", ["Groceries", "Shopping"]),
]

# Nothing in the known list, so any known name here is a wrong match: narration, type, categories.
NEW = [
    (f"UPI-DECATHLON-decathlon@ybl-YESB0000555-{R}223-PAYMENT", "expense", ["Shopping"]),
    (f"UPI-RAPIDO-rapido@ybl-YESB0000556-{R}224-RIDE", "expense", ["Transport"]),
    (f"UPI-RAHUL VERMA-rahul.v@oksbi-SBIN0000066-{R}225-SENT", "expense", ["Transfers"]),
    ("Paid to Priya Sharma", "expense", ["Transfers"]),
    (f"UPI-UDEMY-udemy@icici-ICIC0000123-{R}226-COURSE", "expense", ["Education"]),
    (f"ATW-416021XXXXXX1234-S1AWBL05-BENGALURU-{R}227", "expense", ["Other"]),
]

CASES = [(text, kind) for text, kind, *_ in SAME + TRAPS + NEW]
FIRST_TRAP, FIRST_NEW = len(SAME), len(SAME) + len(TRAPS)


def score(answer: dict) -> dict:
    by_id = {item["id"]: item for item in answer["items"] if isinstance(item.get("id"), int)}
    known = {name.lower() for name in KNOWN_NAMES}

    def merchant(i: int) -> str:
        return str(by_id.get(i, {}).get("merchant", "")).strip().lower()

    misses, merged, wrong_categories = [], [], []
    for i, (text, _, name) in enumerate(SAME):
        if merchant(i) != name.lower():
            misses.append((text[:60], by_id.get(i, {}).get("merchant"), name))

    # A merged name adds this row to the wrong merchant's total, but the category still counts
    # for itself: only a user's correction overrides the LLM's category.
    for i, (text, _, name, categories) in enumerate(TRAPS, FIRST_TRAP):
        if merchant(i) == name.lower():
            merged.append((text[:60], by_id[i].get("merchant"), "different business"))
        if by_id.get(i, {}).get("category") not in categories:
            wrong_categories.append((text[:60], by_id.get(i, {}).get("category"), categories[0]))

    for i, (text, _, categories) in enumerate(NEW, FIRST_NEW):
        if merchant(i) in known:
            merged.append((text[:60], by_id[i].get("merchant"), "not in the known list"))
        if by_id.get(i, {}).get("category") not in categories:
            wrong_categories.append((text[:60], by_id.get(i, {}).get("category"), categories[0]))

    return {
        "same_name": len(SAME) - len(misses),
        "misses": misses,
        "merged": merged,
        "wrong_categories": wrong_categories,
        "missing": len(CASES) - len(by_id.keys() & set(range(len(CASES)))),
    }


def main() -> None:
    args = parse_args("Measure sending known merchant names to the categorizer.", ["app"], 3)
    items = [{"id": i, "text": mask(text), "type": kind} for i, (text, kind) in enumerate(CASES)]

    def make_jobs():
        return [
            {
                "setting": setting,
                "variant": variant,
                "run": run,
                "system": SYSTEM,
                "user": build_prompt(items, names),
                "schema": SCHEMA,
            }
            for setting in args.settings
            for variant, names in (("with names", KNOWN_NAMES), ("without names", []))
            for run in range(args.runs)
        ]

    results = save_or_load(args, make_jobs)
    print(f"\n{len(SAME)} spellings, {len(TRAPS)} traps, {len(NEW)} new. Invented, not a rate on real statements.")
    for variant in dict.fromkeys(r["variant"] for r in results):
        print(f"\n=== {variant}")
        for result in (r for r in results if r["variant"] == variant):
            if not result["answer"]:
                print(f"  failed in {result['seconds']}s: {str(result['error'])[:200]}")
                continue
            s = score(result["answer"])
            print(
                f"  {result['seconds']}s  same name {s['same_name']}/{len(SAME)}  "
                f"merged {len(s['merged'])}/{len(TRAPS) + len(NEW)}  "
                f"wrong categories {len(s['wrong_categories'])}/{len(TRAPS) + len(NEW)}  rows missing {s['missing']}"
            )
            for text, got, expected in s["misses"]:
                print(f"      miss    {text:60s} got {got!r}, wanted {expected!r}")
            for text, got, why in s["merged"]:
                print(f"      merged  {text:60s} got {got!r}: {why}")
            for text, got, expected in s["wrong_categories"]:
                print(f"      WRONG   {text:60s} got {got!r}, wanted {expected!r}")


if __name__ == "__main__":
    main()
