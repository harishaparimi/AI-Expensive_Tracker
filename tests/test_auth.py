"""The POC login: accounts, sessions, and each user's own data."""
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app import db
from app.main import app
from app.models import AuthSession, User

CREDS = {"email": "Asha@Example.com", "password": "correct-horse"}
SWIGGY = {"date": "2026-10-03", "description": "SWIGGY", "amount": 250, "type": "expense", "category": "Food & Dining"}


def new_browser() -> TestClient:
    return TestClient(app)


def test_signup_logs_you_in_and_me_says_who(client):
    assert client.get("/auth/me").json() == {"authenticated": False, "email": None}
    assert client.post("/auth/signup", json=CREDS).status_code == 201
    assert client.get("/auth/me").json() == {"authenticated": True, "email": "asha@example.com"}


def test_a_duplicate_email_is_refused_whatever_its_case(client):
    client.post("/auth/signup", json=CREDS)
    other = new_browser()
    assert other.post("/auth/signup", json={**CREDS, "email": "ASHA@example.com"}).status_code == 409


def test_login_needs_the_right_password(client):
    client.post("/auth/signup", json=CREDS)
    other = new_browser()
    bad = other.post("/auth/login", json={**CREDS, "password": "wrong-password"})
    assert bad.status_code == 401
    unknown = other.post("/auth/login", json={**CREDS, "email": "nobody@example.com"})
    assert unknown.json() == bad.json()  # does not reveal which emails exist
    assert other.post("/auth/login", json=CREDS).status_code == 200


def test_bad_email_or_short_password_is_422(client):
    assert client.post("/auth/signup", json={"email": "nope", "password": "long-enough"}).status_code == 422
    assert client.post("/auth/signup", json={"email": "a@b.co", "password": "short"}).status_code == 422


def test_the_password_and_token_are_not_stored_as_given(client, temp_db):
    client.post("/auth/signup", json=CREDS)
    token = client.cookies.get("session")
    with Session(temp_db) as session:
        assert CREDS["password"] not in session.exec(select(User)).one().password_hash
        assert session.exec(select(AuthSession)).one().token_hash != token


def test_data_follows_the_user_to_another_browser_and_is_not_shared(client):
    client.post("/auth/signup", json=CREDS)
    client.post("/transactions", json=SWIGGY)

    second_device = new_browser()
    assert second_device.get("/transactions").json() == []  # not logged in: its own empty browser
    second_device.post("/auth/login", json=CREDS)
    assert len(second_device.get("/transactions").json()) == 1

    stranger = new_browser()
    stranger.post("/auth/signup", json={"email": "b@example.com", "password": "another-pass"})
    assert stranger.get("/transactions").json() == []


def test_logout_goes_back_to_the_anonymous_browser(client):
    client.post("/transactions", json=SWIGGY)  # anonymous row
    client.post("/auth/signup", json=CREDS)
    assert client.get("/transactions").json() == []  # the user's own, empty
    client.post("/auth/logout")
    assert client.get("/auth/me").json()["authenticated"] is False
    assert len(client.get("/transactions").json()) == 1


def test_a_made_up_or_expired_session_is_ignored(client, temp_db):
    client.cookies.set("session", "not-a-real-token")
    assert client.get("/auth/me").json()["authenticated"] is False

    client.post("/auth/signup", json=CREDS)
    with Session(temp_db) as session:
        row = session.exec(select(AuthSession)).one()
        row.created_at = row.created_at.replace(year=2000)
        session.add(row)
        session.commit()
    assert client.get("/auth/me").json()["authenticated"] is False


def test_the_login_page_is_served(client):
    page = client.get("/login")
    assert page.status_code == 200 and "/auth/" in page.text


def test_name_is_kept_and_optional(client):
    assert client.post("/auth/signup", json={**CREDS, "name": "  Asha Rao "}).json() == {"email": "asha@example.com", "name": "Asha Rao"}
    other = new_browser()
    assert other.post("/auth/login", json=CREDS).json()["name"] == "Asha Rao"
    third = new_browser()
    assert third.post("/auth/signup", json={"email": "b@example.com", "password": "another-pass"}).json()["name"] == ""


def test_login_does_not_enforce_the_signup_password_length(client):
    # Only sign-up sets the rule, so a short wrong password is a 401 and not a confusing 422
    client.post("/auth/signup", json=CREDS)
    assert new_browser().post("/auth/login", json={**CREDS, "password": "x"}).status_code == 401


def test_keep_me_signed_in_off_gives_a_cookie_that_ends_with_the_browser(client):
    client.post("/auth/signup", json=CREDS)
    def session_cookie(response) -> str:  # a new browser also gets a browser_id cookie
        return next(c for c in response.headers.get_list("set-cookie") if c.startswith("session=")).lower()

    assert "max-age" in session_cookie(new_browser().post("/auth/login", json=CREDS))
    assert "max-age" not in session_cookie(new_browser().post("/auth/login", json={**CREDS, "remember": False}))


def test_a_shared_db_from_the_first_login_version_gets_the_name_column(temp_db):
    with temp_db.begin() as connection:
        connection.exec_driver_sql('ALTER TABLE "user" DROP COLUMN name')
    db.init_db()
    with Session(temp_db) as session:
        session.add(User(email="old@example.com", password_hash="x", data_key="k" * 32))
        session.commit()
        assert session.exec(select(User)).one().name == ""
