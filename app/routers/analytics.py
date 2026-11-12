"""Numbers for the dashboard. Each takes ?month=2026-10, or uses the latest month with data."""
from fastapi import APIRouter, Depends, Query
from sqlmodel import Session

from app.db import get_session
from app.models import MonthParam
from app.services import analytics

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
def summary(month: MonthParam = None, session: Session = Depends(get_session)):
    df = analytics.load_transactions(session)
    return analytics.month_totals(df, month or analytics.latest_month(df))


@router.get("/categories")
def categories(month: MonthParam = None, session: Session = Depends(get_session)):
    df = analytics.load_transactions(session)
    month = month or analytics.latest_month(df)
    return {"month": month, "categories": analytics.by_category(df, month)}


@router.get("/trend")
def trend(session: Session = Depends(get_session)):
    return analytics.monthly_trend(analytics.load_transactions(session))


@router.get("/merchants")
def merchants(
    month: MonthParam = None,
    limit: int = Query(default=10, ge=1, le=50),
    session: Session = Depends(get_session),
):
    df = analytics.load_transactions(session)
    month = month or analytics.latest_month(df)
    return {"month": month, "merchants": analytics.top_merchants(df, month, limit)}
