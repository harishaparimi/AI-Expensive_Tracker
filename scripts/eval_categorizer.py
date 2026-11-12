"""How accurate is the categorizer on narrations it has not seen?

Run with:  python -m scripts.eval_categorizer --settings default low off --runs 2 --out answers.json
Re-score:  python -m scripts.eval_categorizer --rescore answers.json   (uses no requests)
Scenarios: --scenario demo sends the demo statement's descriptions exactly as the app does.
           --scenario batches sends the cases below in shuffled batches of 25, one request each.
           Keep running these two: one long mixed list scored 84/84 while the demo batch had Swiggy
           under Transport in 7 of 15 runs. Pass the same scenario with --rescore.

Each run is one real request holding all the cases below. Everything here is invented and hand
labelled, written from general knowledge of how Indian bank and UPI narrations look. No real statement
was used, so these scores are not a measured rate on real statements.
"""
import csv
import random
from collections import Counter
from pathlib import Path

from app.services.categorizer import SCHEMA, SYSTEM, build_prompt
from app.services.llm import mask
from scripts.eval_common import parse_args, save_or_load

R = "612345678"  # the start of an invented reference number, which masking hides

# narration, type, text the merchant name must contain, acceptable categories
CASES = [
    # Food & Dining
    (f"UPI-SWIGGY-swiggy@ybl-YESB0000123-{R}101-PAYMENT", "expense", "swiggy", ["Food & Dining"]),
    (f"UPI/DR/{R}102/ZOMATO LTD/HDFC/zomato.order@hdfcbank/Order", "expense", "zomato", ["Food & Dining"]),
    ("POS 416021XXXXXX1234 DOMINOS PIZZA KORAMAN POS DEBIT", "expense", "domino", ["Food & Dining"]),
    ("Paid to McDonald's", "expense", "mcdonald", ["Food & Dining"]),
    (f"UPI/P2M/{R}103/SRI LAKSHMI TIFFIN CENTRE/PYTM/lakshmitiffin@paytm/Breakfast", "expense", "lakshmi", ["Food & Dining"]),
    ("POS 416021XXXXXX1234 STARBUCKS COFFEE INDIRANAGAR", "expense", "starbucks", ["Food & Dining"]),
    (f"UPI-CHAI POINT-chaipoint@icici-ICIC0000456-{R}104-PAYMENT", "expense", "chai", ["Food & Dining"]),
    (f"UPI-BARBEQUE NATION-bbqnation@axisbank-UTIB0000789-{R}105-DINNER", "expense", "barbeque", ["Food & Dining"]),
    # Groceries
    ("POS 416021XXXXXX1234 DMART AVENUE SUPERMARTS", "expense", "mart", ["Groceries"]),
    (f"UPI-BLINKIT-blinkit@hdfcbank-HDFC0000123-{R}106-PAYMENT", "expense", "blinkit", ["Groceries"]),
    (f"UPI/DR/{R}107/ZEPTO/ICIC/zeptonow@ybl/Order", "expense", "zepto", ["Groceries"]),
    ("Paid to Raju Kirana Stores", "expense", "raju", ["Groceries"]),
    (f"UPI/P2M/{R}108/RELIANCE FRESH/HDFC/relfresh@hdfcbank/Payment", "expense", "reliance", ["Groceries"]),
    (f"UPI-BIGBASKET-bigbasket@axisbank-UTIB0000111-{R}109-PAYMENT", "expense", "bigbasket", ["Groceries"]),
    ("POS 416021XXXXXX1234 MORE RETAIL LTD HSR LAYOUT", "expense", "more", ["Groceries"]),
    # Transport
    (f"UPI-UBER INDIA-uber@axisbank-UTIB0000222-{R}110-PAYMENT", "expense", "uber", ["Transport"]),
    (f"UPI/DR/{R}111/OLA CABS/YESB/olamoney@ybl/Ride", "expense", "ola", ["Transport"]),
    ("Paid to Rapido", "expense", "rapido", ["Transport"]),
    ("POS 416021XXXXXX1234 HP PETROL PUMP MG ROAD", "expense", "hp", ["Transport"]),
    (f"UPI/P2M/{R}112/NAMMA METRO TOP UP/SBIN/bmrcl@sbi/Recharge", "expense", "metro", ["Transport"]),
    (f"UPI-FASTAG RECHARGE-netc.fastag@icici-ICIC0000333-{R}113-TOLL", "expense", "fastag", ["Transport"]),
    # Shopping
    (f"UPI-AMAZON PAY-amazon@apl-UTIB0000444-{R}114-PAYMENT", "expense", "amazon", ["Shopping"]),
    (f"UPI/DR/{R}115/FLIPKART INTERNET/AXIS/flipkart@axisbank/Order", "expense", "flipkart", ["Shopping"]),
    ("Paid to Myntra", "expense", "myntra", ["Shopping"]),
    (f"UPI-AJIO-ajio@ybl-YESB0000555-{R}116-PAYMENT", "expense", "ajio", ["Shopping"]),
    ("POS 416021XXXXXX1234 DECATHLON SPORTS INDIA", "expense", "decathlon", ["Shopping"]),
    ("POS 416021XXXXXX1234 CROMA INFINITI RETAIL", "expense", "croma", ["Shopping"]),
    # Bills & Utilities
    (f"BILLPAY BESCOM ELECTRICITY-{R}117", "expense", "bescom", ["Bills & Utilities"]),
    (f"BIL/BPAY/{R}118/AIRTEL POSTPAID/9000000002", "expense", "airtel", ["Bills & Utilities"]),
    (f"UPI-JIO PREPAID-jio@sbi-SBIN0000666-{R}119-RECHARGE", "expense", "jio", ["Bills & Utilities"]),
    (f"ACH D- ACT FIBERNET BROADBAND-{R}120", "expense", "act", ["Bills & Utilities"]),
    ("Paid to Indane Gas", "expense", "indane", ["Bills & Utilities"]),
    (f"BILLPAY BWSSB WATER BILL-{R}121", "expense", "bwssb", ["Bills & Utilities"]),
    (f"UPI/DR/{R}122/CRED CLUB/HDFC/cred.club@axisb/Credit card bill", "expense", "cred", ["Bills & Utilities", "Transfers", "Other"]),
    # Rent & Housing
    (f"NEFT DR-SBIN0001234-MRS ANITA RAO-RENT-{R}123", "expense", "anita", ["Rent & Housing"]),
    (f"UPI-RAJESH KUMAR-rajesh.k@oksbi-SBIN0000066-{R}124-HOUSE RENT", "expense", "rajesh", ["Rent & Housing"]),
    (f"IMPS/{R}125/GREEN MEADOWS SOCIETY/MAINTENANCE", "expense", "green meadows", ["Rent & Housing", "Bills & Utilities"]),
    (f"ACH D- HDFC BANK HOME LOAN EMI-{R}126", "expense", "hdfc", ["Rent & Housing"]),
    # Entertainment
    (f"UPI-BOOKMYSHOW-bookmyshow@axisbank-UTIB0000777-{R}127-PAYMENT", "expense", "bookmyshow", ["Entertainment"]),
    ("POS 416021XXXXXX1234 PVR INOX FORUM MALL", "expense", "pvr", ["Entertainment"]),
    ("Paid to Wonderla Holidays", "expense", "wonderla", ["Entertainment", "Travel"]),
    # Subscriptions
    (f"UPI-NETFLIX-netflix@icici-ICIC0000888-{R}128-MONTHLY", "expense", "netflix", ["Subscriptions"]),
    (f"UPI-SPOTIFY-spotify@axisbank-UTIB0000999-{R}129-MONTHLY", "expense", "spotify", ["Subscriptions"]),
    (f"ACH D- HOTSTAR SUBSCRIPTION-{R}130", "expense", "hotstar", ["Subscriptions"]),
    (f"UPI/DR/{R}131/GOOGLE PLAY YOUTUBE PREMIUM/UTIB/googleplay@axisbank/Monthly", "expense", "", ["Subscriptions"]),
    (f"UPI-CULT FIT GYM-cultfit@hdfcbank-HDFC0000044-{R}132-MEMBERSHIP", "expense", "cult", ["Subscriptions", "Health"]),
    # Health
    (f"UPI-APOLLO PHARMACY-apollo@hdfcbank-HDFC0000222-{R}133-PAYMENT", "expense", "apollo", ["Health"]),
    ("Paid to Practo Doctor Consult", "expense", "practo", ["Health"]),
    (f"UPI/P2M/{R}134/PHARMEASY/ICIC/pharmeasy@icici/Medicines", "expense", "pharmeasy", ["Health"]),
    ("POS 416021XXXXXX1234 MANIPAL HOSPITALS OPD", "expense", "manipal", ["Health"]),
    (f"UPI-DR LAL PATHLABS-lalpath@ybl-YESB0000444-{R}135-TEST", "expense", "lal", ["Health"]),
    # Education
    (f"UPI-UDEMY-udemy@icici-ICIC0000123-{R}136-COURSE", "expense", "udemy", ["Education"]),
    (f"NEFT DR-UTIB0000123-CHRIST UNIVERSITY-SEM FEES-{R}137", "expense", "christ", ["Education"]),
    (f"UPI/DR/{R}138/COURSERA/HDFC/coursera@hdfcbank/Plus annual", "expense", "coursera", ["Education", "Subscriptions"]),
    # Travel
    (f"UPI-MAKEMYTRIP-mmt@icici-ICIC0000234-{R}139-BOOKING", "expense", "makemytrip", ["Travel"]),
    ("POS 416021XXXXXX1234 INDIGO AIRLINES WEB", "expense", "indigo", ["Travel"]),
    (f"UPI/P2M/{R}140/OYO ROOMS/YESB/oyo@ybl/Stay", "expense", "oyo", ["Travel"]),
    (f"UPI-IRCTC-irctc@sbi-SBIN0000333-{R}141-TICKET", "expense", "irctc", ["Travel", "Transport"]),
    (f"UPI-REDBUS-redbus@axisbank-UTIB0000345-{R}142-TICKET", "expense", "redbus", ["Travel", "Transport"]),
    # Investments
    (f"ACH D- GROWW MUTUAL FUND SIP-{R}143", "expense", "groww", ["Investments"]),
    (f"ACH D- ZERODHA BROKING LTD-{R}144", "expense", "zerodha", ["Investments"]),
    (f"ACH D- ICICI PRUDENTIAL MF SIP-{R}145", "expense", "icici", ["Investments"]),
    (f"TO TRANSFER-PPF ACCOUNT DEPOSIT-{R}146", "expense", "ppf", ["Investments"]),
    (f"ACH D- NPS TRUST CONTRIBUTION-{R}147", "expense", "nps", ["Investments"]),
    (f"ACH D- LIC OF INDIA PREMIUM-{R}148", "expense", "lic", ["Investments", "Bills & Utilities", "Other"]),
    # Transfers
    (f"UPI-RAHUL VERMA-rahul.v@oksbi-SBIN0000066-{R}149-SENT", "expense", "rahul", ["Transfers"]),
    ("Paid to Priya Sharma", "expense", "priya", ["Transfers"]),
    ("Received from Ankit Gupta", "income", "ankit", ["Transfers"]),
    (f"IMPS/{R}150/ARJUN NAIR/KKBK/Dinner share", "expense", "arjun", ["Transfers"]),
    (f"NEFT CR-KKBK0000123-SUNITA VERMA-{R}151", "income", "sunita", ["Transfers", "Income"]),
    (f"UPI/CR/{R}152/MEERA IYER/SBIN/meera.iyer@okaxis/Trip money", "income", "meera", ["Transfers"]),
    # Income
    (f"ACH C- NIMBUS SOFTECH PVT LTD SALARY MAY-{R}153", "income", "nimbus", ["Income"]),
    (f"NEFT CR-CITI0000001-ACME TECHNOLOGIES-SALARY OCT-{R}154", "income", "acme", ["Income"]),
    ("INT.PD 01-07-2026 TO 30-09-2026", "income", "", ["Income"]),
    (f"ACH C- TATA STEEL DIVIDEND-{R}155", "income", "tata", ["Income", "Investments"]),
    (f"CMS/INCOME TAX REFUND/CPC/{R}156", "income", "", ["Income"]),
    (f"UPI/CR/{R}157/UPWORK ESCROW/HDFC/upwork@hdfcbank/Freelance", "income", "upwork", ["Income"]),
    # Refunds: money back gets the category of what was bought, so it reduces that spending.
    (f"UPI-AMAZON PAY-amazon@apl-UTIB0000444-{R}158-REFUND", "income", "amazon", ["Shopping"]),
    (f"UPI/CR/{R}159/SWIGGY/YESB/swiggy@ybl/Refund order", "income", "swiggy", ["Food & Dining"]),
    (f"NEFT CR-HDFC0000240-MYNTRA DESIGNS PVT LTD-REFUND-{R}170", "income", "myntra", ["Shopping"]),
    (f"UPI-UBER INDIA-uber@axisbank-AXIS0000303-{R}171-REVERSAL", "income", "uber", ["Transport"]),
    (f"UPI/CR/{R}172/APOLLO PHARMACY/HDFC/apollo@hdfcbank/Order cancelled", "income", "apollo", ["Health"]),
    # Other
    (f"ATW-416021XXXXXX1234-S1AWBL05-BENGALURU-{R}160", "expense", "atm", ["Other"]),
    (f"NWD-416021XXXXXX1234-ATM WDL-KORAMANGALA-{R}161", "expense", "atm", ["Other"]),
    ("ANNUAL DEBIT CARD FEE INCL GST", "expense", "", ["Other", "Bills & Utilities"]),
    ("SMS ALERT CHARGES JUL-SEP", "expense", "", ["Other", "Bills & Utilities"]),
    # No way to tell who was paid, which the model should admit.
    (f"UPI-PAYTMQR{R}162-paytmqr{R}162@paytm-PYTM0123456-{R}163-PAYMENT", "expense", "", ["Other", "Transfers", "Shopping", "Food & Dining"]),
]


DEMO_STATEMENT = Path(__file__).resolve().parent.parent / "samples" / "demo_statement.csv"
# Text in a demo narration, the text its merchant name must contain, and its category.
DEMO_EXPECTED = [
    ("AIRTEL", "airtel", "Bills & Utilities"),
    ("AMAZON", "amazon", "Shopping"),
    ("APOLLO", "apollo", "Health"),
    ("BIGBASKET", "bigbasket", "Groceries"),
    ("BESCOM", "bescom", "Bills & Utilities"),
    ("BLINKIT", "blinkit", "Groceries"),
    ("BOOKMYSHOW", "bookmyshow", "Entertainment"),
    ("MYNTRA", "myntra", "Shopping"),
    ("SWIGGY", "swiggy", "Food & Dining"),
    ("THIRD WAVE", "thirdwave", "Food & Dining"),
    ("ZOMATO", "zomato", "Food & Dining"),
    ("UBER", "uber", "Transport"),
    ("RAPIDO", "rapido", "Transport"),
    ("NETFLIX", "netflix", "Subscriptions"),
    ("SPOTIFY", "spotify", "Subscriptions"),
    ("HOTSTAR", "hotstar", "Subscriptions"),
    ("CULT", "cult", "Subscriptions"),
    ("GROWW", "groww", "Investments"),
    ("LANDLORD", "landlord", "Rent & Housing"),
    ("NIMBUS", "nimbus", "Income"),
    ("ATW-", "atm", "Other"),
    ("PRIYA", "priya", "Transfers"),
    ("RAHUL", "rahul", "Transfers"),
]
BATCH_SIZE = 25


def demo_cases() -> list[tuple]:
    """Each different description in the demo statement, in the order the app sends them."""
    lines = DEMO_STATEMENT.read_text(encoding="utf-8").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("Date,"))
    seen = {}
    for row in csv.DictReader(lines[header:]):
        if row["Narration"]:
            kind = "expense" if row["Withdrawal Amt."] else "income"
            seen.setdefault((mask(row["Narration"]), kind))
    cases = []
    for text, kind in seen:
        _, merchant_part, category = next(entry for entry in DEMO_EXPECTED if entry[0] in text.upper())
        cases.append((text, kind, merchant_part, [category]))
    return cases


def shuffled_batches() -> list[list[tuple]]:
    cases = list(CASES)
    random.Random(7).shuffle(cases)  # the same batches every time, so runs can be compared
    return [cases[start : start + BATCH_SIZE] for start in range(0, len(cases), BATCH_SIZE)]


def score(answer: dict, cases: list[tuple] = CASES) -> dict:
    by_id = {item["id"]: item for item in answer["items"] if isinstance(item.get("id"), int)}
    wrong, per_category = [], Counter()
    right_category = right_merchant = right_both = 0
    for i, (text, _, merchant_part, categories) in enumerate(cases):
        item = by_id.get(i, {})
        category_ok = item.get("category") in categories
        merchant = str(item.get("merchant", "")).lower().replace(" ", "")
        merchant_ok = i in by_id and merchant_part.replace(" ", "") in merchant
        right_category += category_ok
        right_merchant += merchant_ok
        right_both += category_ok and merchant_ok
        per_category[categories[0]] += category_ok
        if not (category_ok and merchant_ok):
            wrong.append((text[:60], item.get("merchant"), item.get("category"), categories[0]))
    return {
        "category": right_category,
        "merchant": right_merchant,
        "both": right_both,
        "missing": len(cases) - len(by_id.keys() & set(range(len(cases)))),
        "wrong": wrong,
        "per_category": per_category,
    }


def main() -> None:
    args = parse_args(
        "Score the categorizer on invented narrations.", ["default", "low", "off"], 1, ["all", "demo", "batches"]
    )
    batches = {"all": lambda: [CASES], "demo": lambda: [demo_cases()], "batches": shuffled_batches}[args.scenario]()

    def make_jobs():
        jobs = []
        for setting in args.settings:
            for run in range(args.runs):
                for number, cases in enumerate(batches):
                    items = [{"id": i, "text": mask(case[0]), "type": case[1]} for i, case in enumerate(cases)]
                    user = build_prompt(items, [])
                    jobs.append({"setting": setting, "run": run, "batch": number, "system": SYSTEM, "user": user, "schema": SCHEMA})
        return jobs

    results = save_or_load(args, make_jobs)
    print(f"\n{sum(map(len, batches))} invented narrations, hand labelled. Not a measured rate on real statements.")
    for setting in dict.fromkeys(r["setting"] for r in results):
        print(f"\n=== {setting}")
        for result in (r for r in results if r["setting"] == setting):
            if not result["answer"]:
                print(f"  failed in {result['seconds']}s: {str(result['error'])[:200]}")
                continue
            cases = batches[result.get("batch", 0)]
            total = len(cases)
            s = score(result["answer"], cases)
            print(
                f"  run {result['run'] + 1} batch {result.get('batch', 0) + 1}: {result['seconds']}s  "
                f"category {s['category']}/{total}  merchant {s['merchant']}/{total}  both {s['both']}/{total}  "
                f"rows missing {s['missing']}"
            )
            for text, merchant, category, expected in s["wrong"]:
                print(f"      {text:60s} got {merchant!r} / {category!r}, expected {expected!r}")


if __name__ == "__main__":
    main()
