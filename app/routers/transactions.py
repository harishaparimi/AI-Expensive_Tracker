"""Log, list, correct and delete transactions."""
import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from app.db import get_session
from app.models import (
    UNCATEGORIZED,
    Transaction,
    TransactionCreate,
    TransactionRestore,
    TransactionType,
    TransactionUpdate,
)
from app.services.categorizer import categorize, learn_category

router = APIRouter(prefix="/transactions", tags=["transactions"])


def _get_or_404(transaction_id: int, session: Session) -> Transaction:
    transaction = session.get(Transaction, transaction_id)
    if transaction is None:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    return transaction


@router.post("", response_model=Transaction, status_code=201)
def create_transaction(data: TransactionCreate, session: Session = Depends(get_session)):
    transaction = Transaction(**data.model_dump(exclude={"category"}))
    if data.category:
        transaction.category = data.category
        transaction.category_source = "user"
        transaction.merchant = data.description
    else:
        categorize([transaction], session)  # if the LLM fails, it stays Uncategorized
    session.add(transaction)
    session.commit()
    session.refresh(transaction)
    return transaction


@router.get("", response_model=list[Transaction])
def list_transactions(
    start: dt.date | None = None,
    end: dt.date | None = None,
    category: str | None = None,
    type: TransactionType | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
):
    query = select(Transaction)
    if start:
        query = query.where(Transaction.date >= start)
    if end:
        query = query.where(Transaction.date <= end)
    if category:
        query = query.where(Transaction.category == category)
    if type:
        query = query.where(Transaction.type == type)
    query = query.order_by(Transaction.date.desc(), Transaction.id.desc()).offset(offset).limit(limit)
    return session.exec(query).all()


@router.post("/restore", response_model=Transaction, status_code=201)
def restore_transaction(data: TransactionRestore, session: Session = Depends(get_session)):
    """Undo a delete. It gets a new id, which is fine because the list sorts by date first."""
    transaction = Transaction(**data.model_dump())
    session.add(transaction)
    session.commit()
    session.refresh(transaction)
    return transaction


@router.post("/recategorize")
def recategorize(session: Session = Depends(get_session)):
    """Try the LLM again for rows left Uncategorized, for example after a rate limit."""
    rows = session.exec(select(Transaction).where(Transaction.category == UNCATEGORIZED)).all()
    warning = categorize(rows, session) if rows else None
    session.commit()
    return {
        "tried": len(rows),
        "still_uncategorized": sum(t.category == UNCATEGORIZED for t in rows),
        "warning": warning,
    }


@router.get("/{transaction_id}", response_model=Transaction)
def get_transaction(transaction_id: int, session: Session = Depends(get_session)):
    return _get_or_404(transaction_id, session)


@router.patch("/{transaction_id}", response_model=Transaction)
def update_transaction(
    transaction_id: int, data: TransactionUpdate, session: Session = Depends(get_session)
):
    transaction = _get_or_404(transaction_id, session)
    changes = data.model_dump(exclude_none=True)
    category = changes.pop("category", None)
    for field, value in changes.items():
        setattr(transaction, field, value)
    if category:
        learn_category(transaction, category, session)
    session.add(transaction)
    session.commit()
    session.refresh(transaction)
    return transaction


@router.delete("/{transaction_id}", status_code=204)
def delete_transaction(transaction_id: int, session: Session = Depends(get_session)):
    session.delete(_get_or_404(transaction_id, session))
    session.commit()
