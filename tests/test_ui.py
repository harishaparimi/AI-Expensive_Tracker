"""The dashboard page is served, and runs on real data."""
from app.main import BROWSER_COOKIE


def test_the_page_is_served_and_gives_the_browser_its_cookie(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert BROWSER_COOKIE in response.cookies


def test_the_page_has_no_sample_data_or_fake_answers_left(client):
    page = client.get("/").text
    for leftover in ("DEMO", "guessCategory", "KEYWORDS", "SHOW_STATE_PREVIEW", "/api/"):
        assert leftover not in page
