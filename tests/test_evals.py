"""The eval scores decide which LLM settings the app uses, so the scoring itself must be right."""
import pytest

from scripts import eval_known_merchants as known
from scripts.eval_categorizer import CASES, demo_cases, shuffled_batches
from scripts.eval_categorizer import score as score_categorizer
from scripts.eval_insights import (
    deficit_claim_wrong,
    false_overspending_claim,
    wrong_months_of_cover,
    false_budget_claims,
    is_traceable,
    mislabelled_sentences,
    numbers_in,
    quotes,
    rupee_figures,
    text_cut_off,
    wrong_month_figures,
    yearly_cost_stated,
)

AMAZON = {"merchant": "AMAZON PAY", "by_month": {"2026-05": 2814.0, "2026-10": 6849.0}}


def test_rupee_figures_read_indian_grouping_and_lakh_or_crore():
    text = "Rs 3,23,572 and Rs. 500, ₹1,200.50, Rs 78 lakh and Rs 1.17 crore"
    assert rupee_figures(text) == [(323572.0, False), (500.0, False), (1200.5, False), (7800000.0, True), (11700000.0, True)]


def test_numbers_in_finds_every_number_and_ignores_sign_and_booleans():
    assert numbers_in({"a": [1.5, {"b": -2}], "c": "text", "d": True, "e": {}, "f": []}) == {1.5, 2.0}


@pytest.mark.parametrize(
    "value, rounded, expected",
    [(100.4, False, True), (102.0, False, False), (7812345.0, True, True), (7000000.0, True, False)],
)
def test_figures_match_to_the_rupee_or_within_one_percent_if_rounded(value, rounded, expected):
    assert is_traceable(value, rounded, {100.0, 7800000.0}) is expected


def test_quotes_finds_a_stated_figure():
    assert quotes("Aim for Rs 3,23,572 in total.", 323572.0)
    assert not quotes("Aim for three months of spending.", 323572.0)


def test_yearly_cost_must_be_stated_for_the_top_recurring_payment():
    recurring = [{"merchant": "CULT FIT", "yearly_cost": 17988.0}]
    stated = {"merchant": "CULT FIT", "finding": "It costs Rs 17,988 a year.", "action": "Pause it."}
    left_out = {"merchant": "CULT FIT", "finding": "It costs Rs 1,499 a month.", "action": "Pause it."}
    other = {"merchant": "NETFLIX", "finding": "Rs 17,988 a year.", "action": ""}
    assert yearly_cost_stated([stated], recurring)
    assert not yearly_cost_stated([left_out], recurring)
    assert not yearly_cost_stated([other], recurring)
    assert yearly_cost_stated([], [])


@pytest.mark.parametrize(
    "summary, spending, wrong",
    [
        ("October spending exceeded income.", 63183, True),
        ("October spending exceeded income.", 70000, False),
        ("Spending rose to Rs 63,183, passing income in the latest month.", 63183, True),
        ("Spending went above your income.", 63183, True),
        ("Spending stayed below income in October.", 63183, False),
        ("Money left over fell to minus Rs 1,683.", 63183, False),
    ],
)
def test_overspending_claims_are_checked_against_the_totals(summary, spending, wrong):
    assert false_overspending_claim(summary, [{"income": 65000, "spending": spending}]) is wrong


def test_a_merchant_figure_given_to_its_category_is_mislabelled():
    cut = {"merchant": "AMAZON PAY", "finding": "Shopping rose from Rs 2,814 in May to Rs 6,849 in October."}
    assert len(mislabelled_sentences(cut, AMAZON)) == 1


@pytest.mark.parametrize(
    "finding",
    [
        "AMAZON PAY rose from Rs 2,814 in May. Shopping is over its budget.",  # the category has no merchant figure
        "Amazon rose from Rs 2,814 in May, and Shopping is over its budget.",  # the merchant is named
        "Shopping is over its Rs 5,000 budget.",  # a real budget figure
    ],
)
def test_correctly_labelled_findings_are_not_flagged(finding):
    assert mislabelled_sentences({"merchant": "AMAZON PAY", "finding": finding}, AMAZON) == []


def test_a_figure_against_the_wrong_month_is_found():
    wrong = {"merchant": "AMAZON PAY", "finding": "It was Rs 6,849 in May and Rs 6,849 in October."}
    assert wrong_month_figures(wrong, AMAZON) == ["AMAZON PAY: Rs 6,849 in May, actual 2814.0"]
    right = {"merchant": "AMAZON PAY", "finding": "It rose from Rs 2,814 in May to Rs 6,849 in October, and Rs 0 in June."}
    assert wrong_month_figures(right, AMAZON) == []


def test_false_over_budget_claims_are_found():
    budgets = [{"category": "Shopping", "status": "over"}, {"category": "Transport", "status": "ok"}]
    cuts = [
        {"merchant": "MYNTRA", "finding": "MYNTRA rose. Shopping is over its budget."},
        {"merchant": "UBER", "finding": "UBER rose. Transport is over its budget."},
    ]
    assert false_budget_claims(cuts, budgets) == ["UBER: Transport is ok"]


@pytest.mark.parametrize(
    "advice, average, wrong",
    [
        ("Your average money left over is below zero, so build this first.", 7238, True),
        ("Fixing spending comes before investing.", 7238, True),
        ("Build the fund first. The target is Rs 3,23,572.", 7238, False),
        ("Its 1-year return is negative.", 7238, False),
        ("Fixing spending comes before investing.", -500, False),
        ("Build the fund first.", -500, True),
    ],
)
def test_a_claim_about_a_deficit_must_match_the_average(advice, average, wrong):
    assert deficit_claim_wrong([advice], average) is wrong


def test_a_wrong_length_for_the_emergency_target_is_found():
    assert wrong_months_of_cover("About three months of essential spending, or Rs 3,23,572.") == ["three"]
    assert wrong_months_of_cover("This is six months of spending, or 6 months of expenses.") == []
    assert wrong_months_of_cover("Money ran short over the last three months.") == []


def test_a_pasted_copy_is_told_apart_from_other_extra_text():
    assert text_cut_off(['Fine.\n\n{"summary": "Fine."}', "Fine.\n\nspending_cuts:\n- merchant: X"]) == (2, 0)
    assert text_cut_off(["Fine.\n\nMoney left over is Rs 100.", "Just one paragraph."]) == (0, 1)


def test_every_demo_description_has_an_expected_category():
    cases = demo_cases()
    assert len(cases) == 30
    assert all("[NUM]" in text or "[ID]" in text for text, *_ in cases)


def test_shuffled_batches_cover_every_case_once():
    batches = shuffled_batches()
    assert sorted(case[0] for batch in batches for case in batch) == sorted(case[0] for case in CASES)
    assert max(map(len, batches)) <= 25
    assert batches == shuffled_batches()


def test_categorizer_score_counts_right_wrong_and_missing_rows():
    items = [
        {"id": i, "merchant": merchant_part or "Unknown", "category": categories[0]}
        for i, (_, _, merchant_part, categories) in enumerate(CASES)
    ]
    perfect = score_categorizer({"items": items})
    assert (perfect["category"], perfect["merchant"], perfect["both"], perfect["missing"]) == (len(CASES), len(CASES), len(CASES), 0)

    items[0] = {**items[0], "category": "Other"}  # a wrong category
    items[1] = {**items[1], "merchant": "Someone Else"}  # a wrong merchant
    del items[2]  # a row left out
    flawed = score_categorizer({"items": items})
    assert flawed["both"] == len(CASES) - 3
    assert flawed["missing"] == 1
    assert len(flawed["wrong"]) == 3


def known_merchants_answer(changes=None):
    """A perfect answer to scripts/eval_known_merchants.py, with `changes` applied by row id."""
    items = [{"id": i, "merchant": name, "category": "Food & Dining"} for i, (_, _, name) in enumerate(known.SAME)]
    for i, (_, _, _, categories) in enumerate(known.TRAPS, known.FIRST_TRAP):
        items.append({"id": i, "merchant": f"Other {i}", "category": categories[0]})
    for i, (_, _, categories) in enumerate(known.NEW, known.FIRST_NEW):
        items.append({"id": i, "merchant": f"Other {i}", "category": categories[0]})
    for row_id, change in (changes or {}).items():
        items[row_id].update(change)
    return {"items": items}


def test_known_merchants_perfect_answer_scores_full_marks():
    s = known.score(known_merchants_answer())
    assert s["same_name"] == len(known.SAME)
    assert (s["merged"], s["wrong_categories"], s["misses"], s["missing"]) == ([], [], [], 0)


def test_known_merchants_a_different_spelling_counts_as_a_miss_but_case_does_not():
    assert known.score(known_merchants_answer({0: {"merchant": "Bundl"}}))["same_name"] == len(known.SAME) - 1
    assert known.score(known_merchants_answer({0: {"merchant": "SWIGGY"}}))["same_name"] == len(known.SAME)


def test_known_merchants_a_merged_trap_with_a_right_category_is_only_a_merge():
    trap = known.TRAPS[0]
    s = known.score(known_merchants_answer({known.FIRST_TRAP: {"merchant": trap[2]}}))
    assert len(s["merged"]) == 1 and s["wrong_categories"] == []


def test_known_merchants_a_wrong_category_counts_even_when_the_name_also_merged():
    trap = known.TRAPS[0]
    change = {"merchant": trap[2], "category": "Food & Dining"}
    s = known.score(known_merchants_answer({known.FIRST_TRAP: change}))
    assert len(s["merged"]) == 1 and len(s["wrong_categories"]) == 1


def test_known_merchants_a_known_name_on_a_new_business_counts_as_a_merge():
    s = known.score(known_merchants_answer({known.FIRST_NEW: {"merchant": known.KNOWN_NAMES[0]}}))
    assert len(s["merged"]) == 1


def test_known_merchants_every_trap_name_is_really_in_the_known_list():
    assert {name for _, _, name, _ in known.TRAPS} <= set(known.KNOWN_NAMES)
    assert {name for _, _, name in known.SAME} <= set(known.KNOWN_NAMES)
