"""Sign up, log in, log out. The login page calls these with fetch()."""
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse

from app.models import LoginRequest, SignupRequest
from app.services import auth
from app.services.auth import SESSION_DAYS

router = APIRouter(tags=["auth"])

SESSION_COOKIE = "session"
LOGIN_PAGE = Path(__file__).resolve().parent.parent / "ui" / "login.html"


def _set_cookie(request: Request, response: Response, token: str, remember: bool) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_DAYS * 24 * 60 * 60 if remember else None,  # None: gone when the browser closes
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )


@router.get("/login", include_in_schema=False)
def login_page():
    return FileResponse(LOGIN_PAGE, headers={"Cache-Control": "no-cache"})


@router.post("/auth/signup", status_code=201)
def signup(body: SignupRequest, request: Request, response: Response):
    try:
        user, token = auth.signup(body.email, body.password, body.name)
    except auth.AuthError as exc:
        raise HTTPException(409, str(exc))
    _set_cookie(request, response, token, body.remember)
    return {"email": user.email, "name": user.name}


@router.post("/auth/login")
def login(body: LoginRequest, request: Request, response: Response):
    try:
        user, token = auth.login(body.email, body.password)
    except auth.AuthError as exc:
        raise HTTPException(401, str(exc))
    _set_cookie(request, response, token, body.remember)
    return {"email": user.email, "name": user.name}


@router.post("/auth/logout")
def logout(request: Request, response: Response):
    auth.logout(request.cookies.get(SESSION_COOKIE, ""))
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@router.get("/auth/me")
def me(request: Request):
    """Who is logged in. Always 200, so the page can ask without an error in the console."""
    email = getattr(request.state, "user_email", None)
    return {"authenticated": email is not None, "email": email}
