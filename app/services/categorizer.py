"""The LLM finds each transaction's merchant and category."""
import json

from sqlmodel import Session, select

from app import db
from app.models import (
    CATEGORIES,
    UNCATEGORIZED,
    UNKNOWN_MERCHANT,
    DescriptionAnswer,
    MerchantCategory,
    Transaction,
)
from app.services.llm import LLMError, ask_llm, mask

BATCH_SIZE = 100
MAX_KNOWN_NAMES = 200  # a long history must not make every request token heavy
# Payees here are usually people. Their names are left out, because free models may log prompts.
PEOPLE_CATEGORIES = ("Transfers", "Rent & Housing")

SYSTEM = (
    "You categorize transactions from Indian bank statements and UPI apps. "
    "For each item, give the merchant or payee as a short clean name (for example 'Swiggy', "
    "not the full narration) and choose one category. "
    f"If no merchant can be identified, use '{UNKNOWN_MERCHANT}'. "
    # Every category is defined. With only some defined, the model filed the rest under one of
    # them: Swiggy as Transport in 7 of 15 runs, and everything as Shopping once only that was added.
    "Choose the category by what the money was for. "
    "Use 'Food & Dining' for restaurants, cafes and food delivery. "
    "Use 'Groceries' for supermarkets and grocery delivery. "
    "Use 'Shopping' for online stores and buying clothes, electronics and other goods. "
    "Use 'Bills & Utilities' for electricity, water, gas, phone and internet. "
    "Use 'Health' for pharmacies, doctors and hospitals. "
    "Use 'Entertainment' for movies, events and games. "
    "Use 'Education' for fees, courses and coaching. "
    "Use 'Rent & Housing' for rent, even when it is paid to a person. "
    "Use 'Transport' for daily travel such as cabs, autos, metro, fuel and tolls, and 'Travel' for "
    "trips such as flights, hotels and long-distance trains and buses. "
    "Use 'Subscriptions' for recurring streaming and memberships, such as Netflix, Spotify or a gym. "
    "Use 'Transfers' for other money sent to or received from a person, 'Investments' for SIPs and "
    "mutual funds, 'Income' for salary, interest and other earnings, and merchant 'ATM' with "
    "category 'Other' for cash withdrawals. "
    "Money received from a person is 'Transfers', not 'Income', unless the text says it is salary "
    "or payment for work. "
    "A refund, reversal or cashback from a business gets the category of what was bought, not "
    "'Income'. Tax refunds are 'Income'. "
    "Use 'Other' only when none of these fits. "
    "known_merchants lists names already in use. If an item is the same business as one of them, "
    "copy that name exactly, however the text spells it. A different service from the same company "
    "is a different merchant, and so is a business that only shares a word with a known name, so "
    "give those their own name: Google Pay is not Google One, and Tata Motors is not Tata Steel. "
    "Return one entry for every id. Numbers and IDs are masked as [NUM] and [ID]; ignore them."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "merchant": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                },
                "required": ["id", "merchant", "category"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def build_prompt(items: list[dict], known_names: list[str]) -> str:
    """The message sent to the LLM. The evals use this too, so they test what the app really sends."""
    return json.dumps({"known_merchants": known_names, "items": items}, ensure_ascii=False)


def _names_to_reuse(known: dict) -> list[str]:
    """Merchant names the LLM should keep using, newest last, so a long history keeps the recent ones."""
    names = [m.name for m in known.values() if m.category not in PEOPLE_CATEGORIES]
    return names[-MAX_KNOWN_NAMES:]


def categorize(transactions: list[Transaction], session: Session) -> str | None:
    """Fill in merchant and category. Returns a warning if the LLM could not finish."""
    # Rows differing only by reference number mask to the same text, so each is sent once.
    groups: dict[tuple[str, str], list[Transaction]] = {}
    for transaction in transactions:
        groups.setdefault((mask(transaction.description), transaction.type), []).append(transaction)

    known = {m.key: m for m in session.exec(select(MerchantCategory)).all()}
    with Session(db.shared_engine) as shared:
        answered = {(a.text, a.type): a.model_dump() for a in shared.exec(select(DescriptionAnswer)).all()}

    # A description answered before, in any file, needs no request.
    unseen = []
    for key, group in groups.items():
        if key in answered:
            _apply(group, answered[key], known, session)
        else:
            unseen.append(key)

    for start in range(0, len(unseen), BATCH_SIZE):
        batch = unseen[start : start + BATCH_SIZE]
        items = [{"id": i, "text": text, "type": kind} for i, (text, kind) in enumerate(batch)]
        try:
            # Answers are remembered per description below. Saving the whole batch as well would
            # only replay an answer that left rows out, so a retry could never fill them in.
            answer = ask_llm(SYSTEM, build_prompt(items, _names_to_reuse(known)), SCHEMA, use_saved=False)
        except LLMError as exc:
            return f"Some transactions stayed {UNCATEGORIZED}: {exc}"

        # Committed after each batch, so no lock on the shared database waits on the next request.
        with Session(db.shared_engine) as shared:
            for item in answer["items"]:
                if isinstance(item.get("id"), int) and 0 <= item["id"] < len(batch):
                    key = batch[item["id"]]
                    _remember(key, item, shared)
                    _apply(groups[key], item, known, session)
            shared.commit()
    return None


def _remember(key: tuple[str, str], item: dict, shared: Session) -> None:
    """Save a usable answer. A bad one is left out, so the next import asks again."""
    merchant = str(item.get("merchant", "")).strip()
    if merchant and item.get("category") in CATEGORIES:
        text, kind = key
        shared.merge(DescriptionAnswer(text=text, type=kind, merchant=merchant, category=item["category"]))


def _apply(group: list[Transaction], item: dict, known: dict, session: Session) -> None:
    """Set the answer on a group of rows, keeping the merchant's first spelling."""
    name = str(item.get("merchant", "")).strip() or UNKNOWN_MERCHANT
    category = item.get("category") if item.get("category") in CATEGORIES else "Other"
    source = "llm"

    # "Unknown" covers unrelated rows, so it must not get one shared category.
    if name.lower() != UNKNOWN_MERCHANT.lower():
        saved = known.get(name.lower())
        if saved is None:
            saved = MerchantCategory(key=name.lower(), name=name, category=category, source="llm")
            known[name.lower()] = saved
            session.add(saved)
        name = saved.name
        # Only a user's fix overrides the LLM. Its category is right even when it files Swiggy
        # Instamart under Swiggy (measured in scripts/eval_known_merchants.py), so it must stand.
        if saved.source == "user":
            category, source = saved.category, "user"

    for transaction in group:
        transaction.merchant = name
        transaction.category = category
        transaction.category_source = source


def learn_category(transaction: Transaction, category: str, session: Session) -> None:
    """Apply a user's correction to this row, and to the merchant's other rows and future imports."""
    transaction.category = category
    transaction.category_source = "user"
    if transaction.merchant == UNKNOWN_MERCHANT:
        return

    key = transaction.merchant.lower()
    saved = session.get(MerchantCategory, key)
    if saved:
        saved.category, saved.source = category, "user"
    else:
        session.add(MerchantCategory(key=key, name=transaction.merchant, category=category, source="user"))

    same_merchant = select(Transaction).where(Transaction.merchant == transaction.merchant)
    for other in session.exec(same_merchant).all():
        other.category = category
        other.category_source = "user"
