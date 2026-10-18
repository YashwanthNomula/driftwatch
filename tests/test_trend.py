"""Tests for drift trend over time: trend_series, rendering, CLI."""

import csv
import json
import random

import pytest

from driftwatch.cli import EXIT_DRIFT_FOUND, main
from driftwatch.trend import (
    sparkline, trend_direction, trend_series,
)
from driftwatch import report as report_mod

HEADER = ["age", "income", "plan"]


def make_rows(rng, n, age_shift=0.0,
              plan_weights=(0.5, 0.3, 0.2)):
    rows = []
    for _ in range(n):
        age = max(18, int(rng.gauss(35 + age_shift, 10)))
        income = max(20_000, int(rng.lognormvariate(10.9, 0.5)))
        plan = rng.choices(["basic", "plus", "premium"],
                           weights=plan_weights)[0]
        rows.append([age, income, plan])
    return rows


def write_csv(path, rows, header=HEADER):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture()
def series(tmp_path):
    """Reference + three snapshots with steadily rising age drift."""
    ref = tmp_path / "ref.csv"
    write_csv(ref, make_rows(random.Random(42), 300))
    snaps = []
    for seed, shift in [(101, 0.0), (102, 5.0), (103, 10.0)]:
        p = tmp_path / f"snap{shift:g}.csv"
        write_csv(p, make_rows(random.Random(seed), 300, age_shift=shift))
        snaps.append(str(p))
    return str(ref), snaps


def _by_name(report, name):
    return next(f for f in report.features if f.name == name)


# --- trend_series ---------------------------------------------------------

def test_trend_series_rising_age(series):
    ref, snaps = series
    rep = trend_series(ref, snaps, labels=["w1", "w2", "w3"])
    assert rep.labels == ["w1", "w2", "w3"]
    assert rep.n_ref_rows == 300
    assert rep.snapshot_rows == [300, 300, 300]

    age = _by_name(rep, "age")
    psi = age.psi_series
    assert len(psi) == 3
    assert psi[0] < psi[1] < psi[2]
    assert age.direction == "rising"
    assert age.latest_verdict == "significant drift"


def test_trend_series_stable_features_flat(series):
    ref, snaps = series
    rep = trend_series(ref, snaps)
    income = _by_name(rep, "income")
    assert income.direction == "flat"
    assert income.latest_verdict == "no significant change"
    plan = _by_name(rep, "plan")
    assert plan.direction == "flat"


def test_trend_series_falling(series):
    ref, snaps = series
    rep = trend_series(ref, list(reversed(snaps)))
    age = _by_name(rep, "age")
    assert age.direction == "falling"


def test_trend_labels_default_to_file_stem(series):
    ref, snaps = series
    rep = trend_series(ref, snaps)
    assert rep.labels == ["snap0", "snap5", "snap10"]


def test_trend_label_count_mismatch(series):
    ref, snaps = series
    with pytest.raises(ValueError):
        trend_series(ref, snaps, labels=["only-one"])


def test_trend_needs_at_least_one_snapshot(tmp_path):
    ref = tmp_path / "ref.csv"
    write_csv(ref, make_rows(random.Random(1), 50))
    with pytest.raises(ValueError):
        trend_series(str(ref), [])


def test_trend_missing_column_in_snapshot(tmp_path):
    ref = tmp_path / "ref.csv"
    write_csv(ref, make_rows(random.Random(42), 200))
    s1 = tmp_path / "s1.csv"
    write_csv(s1, make_rows(random.Random(7), 200))
    s2 = tmp_path / "s2.csv"
    write_csv(s2, make_rows(random.Random(8), 200), header=["age", "plan"])
    rep = trend_series(str(ref), [str(s1), str(s2)])
    income = _by_name(rep, "income")
    assert income.psi_series[0] is not None
    assert income.psi_series[1] is None
    assert income.points[1].verdict == "n/a"
    assert income.direction in {"rising", "falling", "flat", "n/a"}


def test_trend_excludes_target(series):
    ref, snaps = series
    rep = trend_series(ref, snaps, exclude=("age",))
    assert {f.name for f in rep.features} == {"income", "plan"}


# --- direction / sparkline units ------------------------------------------

def test_trend_direction_units():
    assert trend_direction([0.01, 0.10, 0.30]) == "rising"
    assert trend_direction([0.30, 0.10, 0.01]) == "falling"
    assert trend_direction([0.010, 0.012, 0.009]) == "flat"
    assert trend_direction([None, None]) == "n/a"
    assert trend_direction([0.5, None]) == "n/a"
    # Gaps don't break the slope estimate.
    assert trend_direction([0.01, None, 0.30]) == "rising"


def test_sparkline_units():
    assert sparkline([0.1, 0.2, 0.3]) == "▁▄▇"
    assert sparkline([0.1, None, 0.3]) == "▁ ▇"
    assert sparkline([0.05, 0.05, 0.05]) == "▁▁▁"
    assert sparkline([None, None]) == "  "


# --- rendering --------------------------------------------------------------

def test_render_trend_terminal(series, capsys):
    ref, snaps = series
    rep = trend_series(ref, snaps, labels=["w1", "w2", "w3"])
    text = report_mod.render_trend_terminal(rep)
    assert "driftwatch trend" in text
    assert "w1" in text and "w3" in text
    assert "age" in text and "rising" in text
    assert "↑" in text and "→" in text


def test_render_trend_json(series):
    ref, snaps = series
    rep = trend_series(ref, snaps, labels=["w1", "w2", "w3"])
    data = json.loads(report_mod.render_trend_json(rep))
    assert data["snapshots"][0]["label"] == "w1"
    age = next(f for f in data["features"] if f["name"] == "age")
    assert len(age["psi"]) == 3
    assert age["direction"] == "rising"
    assert age["latest_verdict"] == "significant drift"
    assert data["n_drifted_latest"] >= 1


def test_render_trend_html(series):
    ref, snaps = series
    rep = trend_series(ref, snaps, labels=["w1", "w2", "w3"])
    html_text = report_mod.render_trend_html(rep)
    assert "driftwatch trend report" in html_text
    assert "age" in html_text and "rising" in html_text


# --- CLI --------------------------------------------------------------------

def test_cli_trend_exit_codes(series, capsys):
    ref, snaps = series
    code = main(["trend", ref, *snaps, "--labels", "w1,w2,w3"])
    assert code == EXIT_DRIFT_FOUND
    out = capsys.readouterr().out
    assert "driftwatch trend" in out
    assert "rising" in out

    # Stable trend: all snapshots from the same distribution -> exit 0.
    stable = []
    for seed in (201, 202, 203):
        p = str(snaps[0]).replace("snap0.csv", f"stable{seed}.csv")
        write_csv(p, make_rows(random.Random(seed), 300))
        stable.append(p)
    code = main(["trend", ref, *stable])
    assert code == 0


def test_cli_trend_writes_json_and_html(series, tmp_path):
    ref, snaps = series
    jp = str(tmp_path / "trend.json")
    hp = str(tmp_path / "trend.html")
    code = main(["trend", ref, *snaps, "--json", jp, "--report", hp])
    assert code == EXIT_DRIFT_FOUND
    data = json.loads(open(jp).read())
    assert data["n_drifted_latest"] >= 1
    html_text = open(hp).read()
    assert "driftwatch trend report" in html_text


def test_cli_trend_missing_file_errors(capsys):
    code = main(["trend", "/nope/missing.csv", "/nope/other.csv"])
    assert code == 1
    assert "error" in capsys.readouterr().err
