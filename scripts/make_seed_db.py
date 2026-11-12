"""Build samples/seed.db, the made-up data every new browser starts with.

Run with:  python -m scripts.make_seed_db

It goes through the real app code using the real LLM: about 3 requests, or fewer once answers are
saved. Rerun it after changing a table or the insights prompt, and read the report it prints,
because that report is the first thing a visitor sees.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from sqlmodel import Session, update

from app import db
from app.models import DEMO_SOURCE, Budget, Transaction
from app.services import insights
from app.services.importer import import_statement

DEMO_STATEMENT = db.SEED_PATH.parent / "demo_statement.csv"
DEMO_BUDGETS = {
    "Food & Dining": 8000,
    "Groceries": 6000,
    "Shopping": 5000,
    "Transport": 3500,
    "Subscriptions": 3000,
    "Entertainment": 1000,
    "Bills & Utilities": 2500,
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console cannot otherwise print every character
    db.init_db()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "seed.db"
        engine = db.sqlite_engine(path)
        db.create_tables(engine, db.BROWSER_TABLES)

        with Session(engine) as session:
            result = import_statement(DEMO_STATEMENT.name, DEMO_STATEMENT.read_bytes(), session)
            print(result)
            if result["warning"] or result["uncategorized"]:
                sys.exit("Not saved: some demo rows were left uncategorized. Run this again.")

            session.add_all(Budget(category=c, monthly_limit=limit) for c, limit in DEMO_BUDGETS.items())
            session.commit()
            report = insights.generate_report(session)
            session.exec(update(Transaction).values(source=DEMO_SOURCE))
            session.commit()

        shutil.copyfile(path, db.SEED_PATH)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
