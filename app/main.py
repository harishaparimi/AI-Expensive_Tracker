"""FastAPI app. Run with: uvicorn app.main:app --reload, then open http://127.0.0.1:8000"""
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

from app.db import init_db
from app.models import CATEGORIES
from app.routers import analytics, auth, budgets, chat, imports, insights, transactions
from app.services.auth import user_for_token

BROWSER_COOKIE = "browser_id"
SESSION_COOKIE = auth.SESSION_COOKIE
COOKIE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
VALID_BROWSER_ID = re.compile(r"[0-9a-f]{32}")
DASHBOARD_PAGE = Path(__file__).resolve().parent / "ui" / "ai-expense-tracker.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Expense Tracker", lifespan=lifespan)
app.include_router(transactions.router)
app.include_router(imports.router)
app.include_router(budgets.router)
app.include_router(analytics.router)
app.include_router(insights.router)
app.include_router(chat.router)
app.include_router(auth.router)


@app.middleware("http")
async def identify_browser(request: Request, call_next):
    """Give each browser a random id in a cookie, which picks its database file. The page itself
    sets the cookie, so the requests it then sends together all share one database. The id becomes
    a file name, so a cookie in any other form is replaced and never used.
    A logged-in user's own database file is used instead, so their data follows them anywhere."""
    browser_id = request.cookies.get(BROWSER_COOKIE, "")
    is_new = not VALID_BROWSER_ID.fullmatch(browser_id)
    if is_new:
        browser_id = secrets.token_hex(16)

    user = user_for_token(request.cookies.get(SESSION_COOKIE, ""))
    request.state.user_email = user.email if user else None
    # Every router reads browser_id, so pointing it at the user's key is all the "backend" they need
    request.state.browser_id = user.data_key if user else browser_id

    response = await call_next(request)
    if is_new:
        response.set_cookie(
            BROWSER_COOKIE,
            browser_id,
            max_age=COOKIE_MAX_AGE_SECONDS,
            httponly=True,  # page scripts cannot read it, so injected text cannot steal it
            samesite="lax",  # other sites cannot send changes using it
            secure=request.url.scheme == "https",
        )
    return response


@app.get("/", include_in_schema=False)
def dashboard():
    """The one page of the app. Everything on it comes from the endpoints below."""
    return FileResponse(DASHBOARD_PAGE, headers={"Cache-Control": "no-cache"})


@app.get("/categories", tags=["transactions"])
def list_categories():
    return CATEGORIES
