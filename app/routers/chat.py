"""Questions about the user's money, answered from their plan and the web."""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.db import get_session
from app.models import ChatRequest
from app.services import chat
from app.services.insights import NoDataError
from app.services.llm import LLMError

router = APIRouter(tags=["chat"])


@router.post("/chat")
def ask(data: ChatRequest, session: Session = Depends(get_session)):
    try:
        return chat.answer_question(data.message, data.history, session)
    except NoDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
