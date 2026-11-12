"""Creates synthetic statements in samples/, so demos never need real bank data.

Run with: python scripts/make_demo_data.py
Everything here is invented. The same seed always produces the same files.

Planted patterns for the insights to find:
- food delivery and coffee keep growing from May to October
- shopping spikes in October
- a gym subscription is paid every month
- monthly savings shrink as spending grows
"""
import calendar
import csv
import datetime as dt
import random
import re
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
MONTHS = [(2026, month) for month in range(5, 11)]  # May to October 2026
OPENING_BALANCE = 85000.00

# name, UPI ID, lowest amount, highest amount, payments per month from May to October
VARIABLE_SPENDS = [
    ("SWIGGY", "swiggy@ybl", 220, 650, [5, 6, 7, 9, 11, 14]),
    ("ZOMATO", "zomato@hdfcbank", 180, 600, [4, 4, 6, 7, 8, 10]),
    ("THIRD WAVE COFFEE", "thirdwave@icici", 250, 550, [3, 4, 4, 5, 6, 8]),
    ("BLINKIT", "blinkit@hdfcbank", 150, 900, [2, 3, 3, 3, 4, 4]),
    ("BIGBASKET", "bigbasket@axisbank", 800, 2200, [2, 2, 2, 2, 2, 2]),
    ("UBER INDIA", "uber@axisbank", 120, 480, [6, 6, 7, 7, 8, 10]),
    ("RAPIDO", "rapido@ybl", 40, 120, [4, 4, 5, 4, 5, 5]),
    ("AMAZON PAY", "amazon@apl", 300, 2500, [2, 2, 1, 2, 3, 5]),
    ("MYNTRA", "myntra@icici", 900, 3500, [0, 1, 0, 1, 1, 3]),
    ("BOOKMYSHOW", "bookmyshow@axisbank", 300, 900, [1, 1, 2, 1, 1, 2]),
    ("APOLLO PHARMACY", "apollo@hdfcbank", 150, 900, [1, 0, 1, 1, 0, 1]),
]
# day of month, narration without the reference number, amount
FIXED_SPENDS = [
    (2, "ACH D- GROWW MUTUAL FUND SIP-{ref}", 3000),
    (3, "NEFT DR-SBIN0001234-MRS DEMO LANDLORD-RENT-{ref}", 18000),
    (5, "UPI-NETFLIX-netflix@icici-ICIC0000011-{ref}-MONTHLY", 649),
    (7, "UPI-SPOTIFY-spotify@axisbank-UTIB0000022-{ref}-MONTHLY", 119),
    (9, "UPI-HOTSTAR-hotstar@ybl-YESB0000033-{ref}-MONTHLY", 299),
    (10, "UPI-CULT FIT GYM-cultfit@hdfcbank-HDFC0000044-{ref}-MEMBERSHIP", 1499),
    (15, "UPI-AIRTEL POSTPAID-airtel@apl-UTIB0000055-{ref}-BILL", 599),
]
# Invented people, for money sent to friends and family.
FRIENDS = [("RAHUL DEMO", "rahul.demo@oksbi"), ("PRIYA SAMPLE", "9000000001@ybl")]


def reference(rng: random.Random) -> str:
    return str(rng.randint(600000000000, 699999999999))


def month_transactions(year: int, month: int, index: int, rng: random.Random) -> list[tuple]:
    """Build one month as (date, narration, debit, credit) rows."""
    days = calendar.monthrange(year, month)[1]
    rows = [
        (
            dt.date(year, month, 1),
            f"ACH C- NIMBUS SOFTECH PVT LTD SALARY {calendar.month_abbr[month].upper()}-{reference(rng)}",
            0,
            65000,
        ),
        (
            dt.date(year, month, 12),
            f"BILLPAY BESCOM ELECTRICITY-{reference(rng)}",
            rng.randint(900, 1800),
            0,
        ),
        (
            dt.date(year, month, 20),
            f"ATW-416021XXXXXX1234-S1AWBL05-BENGALURU-{reference(rng)}",
            rng.choice([2000, 3000]),
            0,
        ),
    ]
    for day, narration, amount in FIXED_SPENDS:
        rows.append((dt.date(year, month, day), narration.format(ref=reference(rng)), amount, 0))

    for name, upi_id, low, high, counts in VARIABLE_SPENDS:
        for _ in range(counts[index]):
            bank_code = f"{upi_id.split('@')[1][:4].upper()}0000{rng.randint(100, 999)}"
            narration = f"UPI-{name}-{upi_id}-{bank_code}-{reference(rng)}-PAYMENT"
            rows.append((dt.date(year, month, rng.randint(1, days)), narration, rng.randint(low, high), 0))

    friend, friend_id = rng.choice(FRIENDS)
    rows.append(
        (
            dt.date(year, month, rng.randint(1, days)),
            f"UPI-{friend}-{friend_id}-SBIN0000066-{reference(rng)}-SENT",
            rng.choice([500, 1000, 1500]),
            0,
        )
    )
    if index % 2 == 1:  # a friend pays back every other month
        friend, friend_id = rng.choice(FRIENDS)
        rows.append(
            (
                dt.date(year, month, rng.randint(1, days)),
                f"UPI-{friend}-{friend_id}-SBIN0000066-{reference(rng)}-RECEIVED",
                0,
                rng.choice([500, 800]),
            )
        )
    return rows


def write_csv(path: Path, rows: list[tuple]) -> None:
    """Write a bank-style export, with account details above the table and a footer below it."""
    balance = OPENING_BALANCE
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["DEMO BANK LTD"])
        writer.writerow(["Account holder:", "Demo User"])
        writer.writerow(["Account number:", "XXXXXX1234"])
        writer.writerow(["Statement from 01/05/2026 to 31/10/2026"])
        writer.writerow([])
        writer.writerow(
            ["Date", "Narration", "Chq./Ref.No.", "Value Dt", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"]
        )
        for date, narration, debit, credit in rows:
            balance += credit - debit
            day = date.strftime("%d/%m/%y")
            writer.writerow(
                [
                    day,
                    narration,
                    re.search(r"\d{12}", narration).group(),
                    day,
                    f"{debit:,.2f}" if debit else "",
                    f"{credit:,.2f}" if credit else "",
                    f"{balance:,.2f}",
                ]
            )
        writer.writerow([])
        writer.writerow(["*** End of statement ***"])


def write_pdf(path: Path, lines: list[str], lines_per_page: int = 48) -> None:
    """Write a plain text PDF by hand, so the demo data needs no extra library."""
    pages = [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)]
    page_ids = [4 + 2 * i for i in range(len(pages))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for page_id, page_lines in zip(page_ids, pages):
        commands = ["BT", "/F1 10 Tf", "14 TL", "40 800 Td"]
        for line in page_lines:
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands += [f"({safe}) Tj", "T*"]
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1", "replace")
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>"
        ).encode()
        objects[page_id + 1] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)

    output = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for object_id in sorted(objects):
        offsets[object_id] = len(output)
        output += b"%d 0 obj\n%s\nendobj\n" % (object_id, objects[object_id])
    xref_start = len(output)
    output += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for object_id in sorted(objects):
        output += b"%010d 00000 n \n" % offsets[object_id]
    output += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref_start)
    path.write_bytes(bytes(output))


def upi_app_lines(rows: list[tuple], rng: random.Random) -> list[str]:
    """Lay out rows like a UPI app statement. This layout is invented, not a real app's."""
    lines = ["Transaction Statement for 90XXXXXX01", "Oct 01, 2026 - Oct 31, 2026", ""]
    lines.append("Date   Transaction Details   Type   Amount")
    for date, narration, debit, credit in rows:
        if "-" not in narration or not narration.startswith("UPI"):
            continue
        name = narration.split("-")[1]
        action = "Paid to" if debit else "Received from"
        lines.append(
            f"{date.strftime('%b %d, %Y')}   {action} {name}   "
            f"{'DEBIT' if debit else 'CREDIT'}   INR {debit or credit:,.2f}"
        )
        lines.append(f"Transaction ID T{rng.randint(10**15, 10**16 - 1)}")
        lines.append(f"UTR No. {reference(rng)}")
    lines += ["", "This is a system generated statement."]
    return lines


def main() -> None:
    rng = random.Random(42)
    SAMPLES.mkdir(exist_ok=True)

    months = [month_transactions(year, month, i, rng) for i, (year, month) in enumerate(MONTHS)]
    all_rows = sorted((row for month in months for row in month), key=lambda row: row[0])
    write_csv(SAMPLES / "demo_statement.csv", all_rows)

    october = sorted(months[-1], key=lambda row: row[0])
    write_pdf(SAMPLES / "demo_upi_statement.pdf", upi_app_lines(october, rng))

    for (year, month), rows in zip(MONTHS, months):
        spent = sum(row[2] for row in rows)
        earned = sum(row[3] for row in rows)
        print(f"{year}-{month:02d}: {len(rows):3d} rows, earned {earned:>7,}, spent {spent:>7,}, saved {earned - spent:>7,}")
    print(f"Wrote {len(all_rows)} rows to samples/demo_statement.csv and October to samples/demo_upi_statement.pdf")


if __name__ == "__main__":
    main()
