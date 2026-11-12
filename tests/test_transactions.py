import pytest

BASE = {"date": "2026-10-03", "description": "Lunch", "amount": 250, "type": "expense"}


def add(client, **changes):
    response = client.post("/transactions", json={**BASE, **changes})
    assert response.status_code == 201, response.text
    return response.json()


def test_create_with_category_uses_no_llm_request(client, fake_llm):
    created = add(client, category="Food & Dining")
    assert (created["category"], created["category_source"], created["source"]) == ("Food & Dining", "user", "manual")
    assert fake_llm.calls == []


def test_create_without_category_asks_the_llm(client, fake_llm):
    created = add(client, description="SWIGGY order")
    assert (created["merchant"], created["category"], created["category_source"]) == ("Swiggy", "Food & Dining", "llm")
    assert len(fake_llm.calls) == 1


def test_undoing_a_delete_brings_the_row_back_as_it_was(client, fake_llm):
    created = add(client, description="SWIGGY order")
    assert client.delete(f"/transactions/{created['id']}").status_code == 204
    requests_used = len(fake_llm.calls)

    response = client.post("/transactions/restore", json=created)
    assert response.status_code == 201, response.text
    assert {**response.json(), "id": None} == {**created, "id": None}
    assert len(fake_llm.calls) == requests_used  # the AI's answer comes back with the row


def test_undo_keeps_what_a_new_row_could_not_have(client, fake_llm):
    created = add(client, category="Shopping")
    client.delete(f"/transactions/{created['id']}")
    row = {**created, "merchant": "MYNTRA", "source": "demo", "category_source": "llm", "dedupe_hash": "abc"}
    restored = client.post("/transactions/restore", json=row).json()
    assert (restored["merchant"], restored["source"], restored["category_source"], restored["dedupe_hash"]) == (
        "MYNTRA",
        "demo",
        "llm",
        "abc",
    )


def test_create_still_works_when_the_llm_fails(client, fake_llm):
    fake_llm.fail = True
    created = add(client)
    assert (created["merchant"], created["category"]) == ("Unknown", "Uncategorized")


def test_identical_manual_entries_are_both_kept(client, fake_llm):
    add(client, category="Food & Dining")
    add(client, category="Food & Dining")
    assert len(client.get("/transactions").json()) == 2


@pytest.mark.parametrize(
    "changes",
    [{"amount": 0}, {"amount": -5}, {"type": "gift"}, {"description": ""}, {"category": "Made Up"}, {"date": "soon"}],
)
def test_invalid_input_is_rejected(client, fake_llm, changes):
    assert client.post("/transactions", json={**BASE, **changes}).status_code == 422
    assert client.get("/transactions").json() == []


def test_list_filters(client, fake_llm):
    add(client, date="2026-09-01", category="Transport", description="Cab")
    add(client, date="2026-10-01", category="Food & Dining", description="Dinner")
    add(client, date="2026-10-02", category="Income", description="Refund", type="income")

    def descriptions(**params):
        return [t["description"] for t in client.get("/transactions", params=params).json()]

    assert descriptions() == ["Refund", "Dinner", "Cab"]  # newest first
    assert descriptions(category="Transport") == ["Cab"]
    assert descriptions(type="income") == ["Refund"]
    assert descriptions(start="2026-10-01", end="2026-10-01") == ["Dinner"]
    assert descriptions(limit=1, offset=1) == ["Dinner"]


def test_update_and_delete(client, fake_llm):
    created = add(client, category="Food & Dining")
    url = f"/transactions/{created['id']}"

    updated = client.patch(url, json={"amount": 300.5, "description": "Team lunch"}).json()
    assert (updated["amount"], updated["description"]) == (300.5, "Team lunch")
    assert client.get(url).json()["amount"] == 300.5

    assert client.delete(url).status_code == 204
    assert client.get(url).status_code == 404


@pytest.mark.parametrize("method", ["get", "delete"])
def test_missing_transaction_is_404(client, method):
    assert getattr(client, method)("/transactions/999").status_code == 404


def test_correcting_a_category_applies_to_the_merchant_and_future_rows(client, fake_llm):
    first = add(client, description="SWIGGY order one")
    second = add(client, description="SWIGGY order two")
    assert first["category"] == second["category"] == "Food & Dining"

    client.patch(f"/transactions/{first['id']}", json={"category": "Groceries"})

    assert client.get(f"/transactions/{second['id']}").json()["category"] == "Groceries"
    future = add(client, description="SWIGGY order three")
    assert (future["category"], future["category_source"]) == ("Groceries", "user")


def test_correcting_an_unknown_merchant_changes_only_that_row(client, fake_llm):
    fake_llm.fail = True
    first = add(client, description="Cash one")
    second = add(client, description="Cash two")
    client.patch(f"/transactions/{first['id']}", json={"category": "Other"})
    assert client.get(f"/transactions/{first['id']}").json()["category"] == "Other"
    assert client.get(f"/transactions/{second['id']}").json()["category"] == "Uncategorized"
