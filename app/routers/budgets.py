"""Monthly budgets per category, and how a month is doing against them."""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.db import get_session
from app.models import Budget, BudgetCategory, BudgetUpdate, MonthParam
from app.services import analytics

router = APIRouter(prefix="/budgets", tags=["budgets"])


@router.get("")
def list_budgets(month: MonthParam = None, session: Session = Depends(get_session)):
    df = analytics.load_transactions(session)
    month = month or analytics.latest_month(df)
    budgets = session.exec(select(Budget)).all()
    return {"month": month, "budgets": analytics.budget_status(df, budgets, month)}


@router.put("/{category}", response_model=Budget)
def set_budget(category: BudgetCategory, data: BudgetUpdate, session: Session = Depends(get_session)):
    budget = session.merge(Budget(category=category, monthly_limit=data.monthly_limit))
    session.commit()
    session.refresh(budget)
    return budget


@router.delete("/{category}", status_code=204)
def delete_budget(category: BudgetCategory, session: Session = Depends(get_session)):
    budget = session.get(Budget, category)
    if budget is None:
        raise HTTPException(status_code=404, detail="No budget set for this category.")
    session.delete(budget)
    session.commit()
