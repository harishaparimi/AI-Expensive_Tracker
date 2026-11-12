"""AI spending insights and investment advice, plus the live fund figures behind them."""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.db import get_session
from app.services import insights, market
from app.services.llm import LLMError

router = APIRouter(tags=["insights"])


@router.get("/market")
def market_snapshot():
    return market.get_funds()


@router.get("/insights")
def latest_insights(session: Session = Depends(get_session)):
    """The last report, so reloading the page uses no LLM request."""
    report = insights.latest_report(session)
    if report is None:
        raise HTTPException(status_code=404, detail="No insights yet.")
    return report


@router.post("/insights")
def create_insights(session: Session = Depends(get_session)):
    try:
        return insights.generate_report(session)
    except insights.NoDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
