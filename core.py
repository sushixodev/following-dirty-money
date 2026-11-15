from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.sparse import coo_matrix

ROOT = Path(__file__).parent
TRAIN_END, BREAK, REVIEWED = 34, 43, 10
TRUTH = {1: "illicit", 0: "licit", -1: "unknown"}
OUTCOME = {(True, 1): "a true positive", (False, 0): "a true negative",
           (False, 1): "a false negative: the model missed it", (True, 0): "a false positive: a false alarm"}

data = np.load(ROOT / "artifacts" / "app_data.npz")
tx_ids, t, y, src, dst = (data[key] for key in ("tx_ids", "t", "y", "src", "dst"))
models = dict(zip(data["model_names"].tolist(), data["model_risk"].astype(np.float32)))
DEPLOYED = data["deployed"].item()
FLAG_LOW, FLAG_HIGH, DRIFT_HIGH = data["limits"].tolist()
risk = models[DEPLOYED]
flagged = risk >= 0.5
health = pd.read_csv(ROOT / "results" / "monitor.csv", index_col=0)
results = pd.read_csv(ROOT / "results" / "test_results.csv", index_col=0)
budget_results = pd.read_csv(ROOT / "results" / "review_budget.csv", index_col=0)

row = {tx: i for i, tx in enumerate(tx_ids.tolist())}
adjacency = coo_matrix((np.ones(2 * len(src)), (np.r_[src, dst], np.r_[dst, src])), shape=(len(t), len(t))).tocsr()
degree = np.diff(adjacency.indptr)
PERIODS = np.arange(TRAIN_END + 1, t.max() + 1)
unseen = (t > TRAIN_END) & (y >= 0)
connected = degree >= 3
EXAMPLES = {"illicit": np.flatnonzero(unseen & connected & (y == 1) & (risk >= 0.8)),
            "licit": np.flatnonzero(unseen & connected & (y == 0) & (risk < 0.2)),
            "error": np.flatnonzero(unseen & (flagged != (y == 1))),
            "random": np.flatnonzero(unseen)}
rng = np.random.default_rng()


def neighbours(i):
    return adjacency[i].indices


def find(text):
    text = (text or "").strip()
    return row.get(int(text)) if text.isdigit() else None


def example(kind):
    return str(tx_ids[rng.choice(EXAMPLES[kind])])


def standing(i):
    same = t == t[i]
    if flagged[i]:
        return f"Riskier than {np.mean(risk[same] < risk[i]):.1%} of the {same.sum():,} transactions in period {t[i]}."
    return f"Less risky than {np.mean(risk[same] > risk[i]):.0%} of the {same.sum():,} transactions in period {t[i]}."


def company(i):
    near = neighbours(i)
    also = int(np.sum(flagged[near]))
    if len(near) == 1:
        return f"Its only linked transaction is {'also flagged' if also else 'not flagged'}."
    if also == 0:
        return f"None of its {len(near)} linked transactions are flagged."
    return f"{also} of its {len(near)} linked transactions {'is' if also == 1 else 'are'} also flagged."


def agreement(i):
    votes = sum(int(p[i] >= 0.5) for p in models.values())
    if votes == len(models):
        return f"All {len(models)} models flag it."
    if votes == 0:
        return f"None of the {len(models)} models flag it."
    return f"{votes} of the {len(models)} models flag{'s' if votes == 1 else ''} it."


def truth(i):
    if y[i] < 0:
        return "Ground truth: unknown, it was never investigated."
    return f"Ground truth: {TRUTH[int(y[i])]}, so this is {OUTCOME[(bool(flagged[i]), int(y[i]))]}."


def explanation(i):
    period = "a period the model never saw" if t[i] > TRAIN_END else "a training period"
    verdict = "Flagged as likely illicit" if flagged[i] else "Not flagged"
    lines = [f"### Transaction {tx_ids[i]}", f"**{verdict}**, risk {risk[i]:.0%} (period {t[i]}, {period}).", ""]
    lines += [f"- {sentence}" for sentence in (standing(i), company(i), agreement(i), truth(i))]
    return "\n".join(lines)


def model_table(i):
    rows = [f"| {name}{' (deployed)' if name == DEPLOYED else ''} | {p[i]:.0%} |" for name, p in models.items()]
    return "\n".join(["| Model | Risk |", "|---|---|", *rows])


def points(nodes, pos, size, outline, colorbar):
    xyz = np.array([pos[n] for n in nodes])
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        marker=dict(size=size, color=risk[nodes], colorscale="Reds", cmin=0, cmax=1, line=dict(color=outline, width=2),
                    colorbar=dict(title="risk", thickness=12, len=0.6) if colorbar else None, showscale=colorbar),
        text=[f"transaction {tx_ids[n]}<br>risk {risk[n]:.0%}<br>links {degree[n]}<br>ground truth {TRUTH[int(y[n])]}"
              for n in nodes],
        hovertemplate="%{text}<extra></extra>")


def bubble_map(i):
    first = neighbours(i)[:40]
    graph = nx.Graph([(i, j) for j in first] + [(j, k) for j in first for k in neighbours(j)[:5] if k != i])
    pos = nx.spring_layout(graph, dim=3, seed=1)
    others = [n for n in graph if n != i]
    lines = np.array([[pos[a], pos[b], [np.nan] * 3] for a, b in graph.edges]).reshape(-1, 3)
    hidden = dict(visible=False, range=[-1.15, 1.15])
    fig = go.Figure([
        go.Scatter3d(x=lines[:, 0], y=lines[:, 1], z=lines[:, 2], mode="lines", line=dict(color="lightgrey", width=2),
                     hoverinfo="skip"),
        points(others, pos, np.clip(5 + 3 * np.sqrt(degree[others]), 5, 22), "grey", True),
        points([i], pos, 26, "black", False),
    ])
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden, aspectmode="cube",
                                 camera=dict(eye=dict(x=1.1, y=1.1, z=0.8))))
    return fig


def linked(i):
    near = sorted(neighbours(i), key=lambda j: -risk[j])[:30]
    return [(f"{tx_ids[j]} (risk {risk[j]:.0%})", str(tx_ids[j])) for j in near]


def inspect(text):
    i = find(text)
    if i is None:
        return "No transaction with that ID in the Elliptic data.", "", None, []
    return explanation(i), model_table(i), bubble_map(i), linked(i)


def queue(period, budget, reveal):
    period, budget = int(period), int(budget)
    members = np.flatnonzero(t == period)
    top = members[np.argsort(-risk[members], kind="stable")][:budget]
    table = pd.DataFrame({
        "Rank": np.arange(1, len(top) + 1),
        "Transaction": tx_ids[top].astype(str),
        "Risk %": np.round(risk[top] * 100).astype(int),
        "Links": degree[top],
        "Flagged links": [int(np.sum(flagged[neighbours(j)])) for j in top],
        "Ground truth": [TRUTH[int(y[j])] if reveal else "?" for j in top],
    })
    illicit = np.sum(y[members] == 1)
    lines = [f"Period {period} has {len(members):,} transactions, {illicit} of them later confirmed illicit. "
             f"The table lists the {budget} the model ranks highest."]
    if reveal:
        lines.append(f"Of these {budget}: {np.sum(y[top] == 1)} illicit, {np.sum(y[top] == 0)} licit, "
                     f"{np.sum(y[top] == -1)} never investigated. {budget} random reviews would find about "
                     f"{budget * illicit / len(members):.1f} illicit transactions.")
    if period >= BREAK:
        lines.append(f"This period is after the dark market shut down in period {BREAK}.")
    return "\n\n".join(lines), table, tx_ids[top].astype(str).tolist()


def health_check(period):
    h = health.loc[int(period)]
    alarm = {signal: bool(h[f"{signal} alarm"]) for signal in ("flag rate", "drift AUC", "analyst hit rate")}
    rows = [
        ("Flag rate (no labels)", f"{h['flag rate']:.1%}", f"{max(FLAG_LOW, 0):.1%} to {FLAG_HIGH:.1%}",
         alarm["flag rate"]),
        ("Data drift AUC vs previous 3 periods (no labels)", f"{h['drift AUC']:.2f}", f"below {DRIFT_HIGH:.2f}",
         alarm["drift AUC"]),
        (f"Analyst hit rate on the {REVIEWED} riskiest", f"{round(h['analyst hit rate'] * REVIEWED)} of {REVIEWED}",
         f"at least {REVIEWED // 2} of {REVIEWED}", alarm["analyst hit rate"]),
    ]
    table = ["| Signal | Value | Normal | Status |", "|---|---|---|---|"]
    table += [f"| {name} | {value} | {normal} | {'alarm' if fired else 'ok'} |" for name, value, normal, fired in rows]
    title = f"### Period {int(period)}: {'alarm' if any(alarm.values()) else 'no alarm'}"
    later = f"F1 for this period, known only months later: {h['true F1']:.2f}."
    return "\n".join([title, "", *table, "", later])


def health_chart(period):
    test = health.loc[PERIODS]
    fig = go.Figure([
        go.Scatter(x=PERIODS, y=test["analyst hit rate"], mode="lines+markers", name="analyst hit rate (top 10)"),
        go.Scatter(x=PERIODS, y=test["true F1"], mode="lines+markers", name="true F1 (known months later)"),
    ])
    fig.add_vline(x=BREAK - 0.5, line_dash="dash", line_color="grey", annotation_text="dark market shut down",
                  annotation_position="bottom left")
    fig.add_vline(x=int(period), line_dash="dot", line_color="black")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="period", dtick=1),
                      yaxis=dict(title="score", range=[-0.05, 1.05]), legend=dict(orientation="h", y=1.15))
    return fig


def health_view(period):
    return health_check(period), health_chart(period)


def about():
    ours, forest = results.loc[DEPLOYED], results.loc["Random forest"]
    missed = np.mean(~flagged[(t > TRAIN_END) & (t < BREAK) & (y == 1)])
    found = budget_results.loc[50]
    return f"""
**Data.** The Elliptic dataset ([Weber et al., 2019](https://arxiv.org/abs/1908.02591)): {len(t):,} Bitcoin transactions
in {t.max()} periods of about two weeks, linked by {len(src):,} payments; {np.sum(y == 1):,} labelled illicit and
{np.sum(y == 0):,} licit, the rest unknown. The models were trained on periods 1-{TRAIN_END} and tested on periods
{PERIODS[0]}-{PERIODS[-1]}.

**Model.** A random forest that also receives a GraphSAGE network's summary of each transaction's neighbourhood
({DEPLOYED}). On the test periods: F1 {ours['f1']:.3f}, precision {ours['precision']:.3f}, recall
{ours['recall']:.3f}. A plain random forest reaches F1 {forest['f1']:.3f}. Reviewing the 50 riskiest transactions in
each test period finds {found[DEPLOYED]} illicit ones; random reviews find about {found['random reviews']}.

**Limitations.** After a large dark market shut down in period {BREAK}, every model stopped working, and only the
analyst hit rate signalled it immediately. In the earlier test periods the model misses {missed:.0%} of illicit
transactions. The features are anonymised, so explanations rely on the graph and on model agreement.

Prototype for a high-school hackathon, not financial advice. Code and analysis:
[github.com/sushixodev/following-dirty-money](https://github.com/sushixodev/following-dirty-money).
"""
