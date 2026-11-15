import numpy as np
import pytest

import core


def first(mask):
    return int(np.flatnonzero(mask)[0])


def tx(i):
    return str(core.tx_ids[i])


def test_graph_is_symmetric_with_no_isolated_transactions():
    assert (core.adjacency != core.adjacency.T).nnz == 0
    assert core.degree.min() >= 1


def test_model_scores_are_probabilities_for_every_transaction():
    assert core.DEPLOYED in core.models
    for scores in core.models.values():
        assert scores.shape == core.t.shape
        assert np.all((scores >= 0) & (scores <= 1))


def test_find_accepts_ids_and_rejects_anything_else():
    assert core.find(f"  {tx(0)} ") == 0
    for text in ("", None, "abc", "12.5", "-3", "999999999999"):
        assert core.find(text) is None


@pytest.mark.parametrize("kind", core.EXAMPLES)
def test_examples_come_from_unseen_labelled_periods_and_match_their_label(kind):
    i = core.find(core.example(kind))
    assert core.t[i] > core.TRAIN_END and core.y[i] >= 0
    if kind == "illicit":
        assert core.y[i] == 1 and core.risk[i] >= 0.8
    elif kind == "licit":
        assert core.y[i] == 0 and core.risk[i] < 0.2
    elif kind == "error":
        assert core.flagged[i] != (core.y[i] == 1)


def test_company_sentence_reads_naturally_for_one_and_many_links():
    assert core.company(first(core.degree == 1)).startswith("Its only linked transaction is")
    busy = first(core.degree > 5)
    assert f"of its {core.degree[busy]} linked transactions" in core.company(busy)


def test_agreement_counts_the_models_with_correct_grammar():
    votes = sum((scores >= 0.5).astype(int) for scores in core.models.values())
    n = len(core.models)
    assert core.agreement(first(votes == 0)) == f"None of the {n} models flag it."
    assert core.agreement(first(votes == n)) == f"All {n} models flag it."
    assert core.agreement(first(votes == 1)) == f"1 of the {n} models flags it."
    assert core.agreement(first(votes == 2)) == f"2 of the {n} models flag it."


def test_truth_names_every_outcome():
    assert core.truth(first(core.y == -1)) == "Ground truth: unknown, it was never investigated."
    outcomes = [(core.flagged & (core.y == 1), "a true positive."),
                (~core.flagged & (core.y == 0), "a true negative."),
                (~core.flagged & (core.y == 1), "a false negative: the model missed it."),
                (core.flagged & (core.y == 0), "a false positive: a false alarm.")]
    for mask, ending in outcomes:
        assert core.truth(first(mask)).endswith(ending)


def test_inspect_returns_everything_the_page_shows():
    i = first(core.flagged & (core.y == 1) & (core.t > core.TRAIN_END))
    explanation, scores, chart, linked = core.inspect(tx(i))
    assert explanation.startswith(f"### Transaction {tx(i)}") and "**Flagged as likely illicit**" in explanation
    assert explanation.count("\n- ") == 4
    assert scores.count("\n| ") == len(core.models) and "(deployed)" in scores
    assert [trace.type for trace in chart.data] == ["scatter3d"] * 3
    assert len(chart.data[2].x) == 1 and tx(i) in chart.data[2].text[0]
    risks = [core.risk[core.find(value)] for _, value in linked]
    assert risks == sorted(risks, reverse=True)
    assert {value for _, value in linked} <= {tx(j) for j in core.neighbours(i)}


def test_inspect_marks_unflagged_transactions():
    explanation = core.inspect(tx(first(~core.flagged & (core.y == 0))))[0]
    assert "**Not flagged**" in explanation


def test_inspect_handles_unknown_ids():
    explanation, scores, chart, linked = core.inspect("not an id")
    assert explanation.startswith("No transaction") and chart is None and linked == []


def test_queue_lists_the_riskiest_transactions_of_the_period():
    summary, table, ids = core.queue(40, 20, False)
    picked = [core.find(value) for value in ids]
    assert len(table) == 20 and all(core.t[i] == 40 for i in picked)
    risks = core.risk[picked]
    assert np.all(np.diff(risks) <= 0)
    rest = np.setdiff1d(np.flatnonzero(core.t == 40), picked)
    assert core.risk[rest].max() <= risks.min()
    assert set(table["Ground truth"]) == {"?"}


def test_queue_reveal_reports_counts_that_add_up():
    summary, table, ids = core.queue(40, 20, True)
    found = table["Ground truth"].value_counts()
    assert found.sum() == 20
    assert f"Of these 20: {found.get('illicit', 0)} illicit, {found.get('licit', 0)} licit" in summary


def test_health_alarm_fires_exactly_when_the_model_breaks():
    assert core.health_check(core.BREAK - 1).startswith(f"### Period {core.BREAK - 1}: no alarm")
    for period in range(core.BREAK, core.PERIODS[-1] + 1):
        assert core.health_check(period).startswith(f"### Period {period}: alarm")


def test_health_chart_covers_every_unseen_period():
    chart = core.health_chart(40)
    assert list(chart.data[0].x) == list(core.PERIODS)


def test_about_quotes_the_notebook_results():
    assert f"{core.results.loc[core.DEPLOYED, 'f1']:.3f}" in core.about()
