"""Upload a bank or UPI statement, or start this browser's data again."""
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from sqlmodel import Session

from app.db import get_session, reset_browser_data
from app.services.importer import StatementError, import_statement
from app.services.llm import LLMError

router = APIRouter(tags=["import"])


@router.post("/import")
def import_file(file: UploadFile, session: Session = Depends(get_session)):
    try:
        return import_statement(file.filename, file.file.read(), session)
    except StatementError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/reset", status_code=204)
def reset(request: Request, demo: bool = True):
    """Delete this browser's data and start again with the demo data, or with ?demo=false, empty."""
    reset_browser_data(request.state.browser_id, demo)
