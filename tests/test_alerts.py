"""Tests for trend alert hooks: escalation detection, hook sinks, dedup."""

import csv
import json
import random

import pytest

from driftwatch.alerts import (
    Alert,
    detect_alerts,
    dispatch_alerts,
    filter_already_reported,
    load_alert_state,
    parse_alert_hook,
    record_alerts,
    save_alert_state,
)
from driftwatch.cli import EXIT_DRIFT_FOUND, main
from driftwatch.trend import TrendFeature, TrendPoint, TrendReport

OK = "no significant change"
WATCH = "moderate shift — watch"
DRIFT = "significant drift"


def _feature(name, points, kind="numeric"):
    """points: list of (psi, verdict, ks)."""
    return TrendFeature(
        name=name, kind=kind,
        points=[TrendPoint(psi=p, verdict=v, ks=k) for p, v, k in points])


def _report(features, labels=("w1", "w2", "w3")):
    return TrendReport(features=features, labels=list(labels),
                       n_ref_rows=100, snapshot_rows=[100] * len(labels))


# --- detect_alerts ----------------------------------------------------------

def test_newly_drifted_feature_alerts():
    rep = _report([_feature("age", [
        (0.011, OK, 0.03), (0.189, WATCH, 0.20), (0.670, DRIFT, 0.33)])])
    alerts = detect_alerts(rep)
    assert len(alerts) == 1
    a = alerts[0]
    assert a.feature == "age"
    assert a.previous_verdict == WATCH
    assert a.new_verdict == DRIFT
    assert a.psi == 0.670
    assert a.ks == 0.33
    assert a.direction == "rising"
    assert a.snapshot_index == 2
    assert a.snapshot_label == "w3"


def test_jump_straight_to_drift_alerts():
    rep = _report([_feature("age", [(0.01, OK, 0.02), (0.50, DRIFT, 0.40)])],
                  labels=("w1", "w2"))
    alerts = detect_alerts(rep)
    assert len(alerts) == 1
    assert alerts[0].previous_verdict == OK
    assert alerts[0].new_verdict == DRIFT


def test_rising_into_watch_alerts():
    rep = _report([_feature("plan", [(0.01, OK, None), (0.15, WATCH, None)])],
                  labels=("w1", "w2"), )
    alerts = detect_alerts(rep)
    assert len(alerts) == 1
    assert alerts[0].new_verdict == WATCH
    assert alerts[0].direction == "rising"


def test_already_drifted_feature_does_not_realert():
    # Drifted in the previous snapshot and still drifted: no new alert.
    rep = _report([_feature("age", [
        (0.01, OK, 0.02), (0.40, DRIFT, 0.35), (0.60, DRIFT, 0.45)])])
    assert detect_alerts(rep) == []
    # Drifted for the whole series: still silent.
    rep = _report([_feature("age", [(0.30, DRIFT, 0.35), (0.50, DRIFT, 0.45)])],
                  labels=("w1", "w2"))
    assert detect_alerts(rep) == []


def test_stable_and_recovering_features_do_not_alert():
    rep = _report([
        _feature("income", [(0.01, OK, 0.02), (0.02, OK, 0.03)]),
        _feature("age", [(0.40, DRIFT, 0.35), (0.01, OK, 0.02)]),
    ], labels=("w1", "w2"))
    assert detect_alerts(rep) == []


def test_watch_without_rising_direction_does_not_alert():
    # PSI flat, verdict worsened only via KS: a blip, not a rising trend.
    rep = _report([_feature("age", [(0.05, OK, 0.02), (0.05, WATCH, 0.15)])],
                  labels=("w1", "w2"))
    assert detect_alerts(rep) == []


def test_drift_alerts_even_when_psi_flat():
    # Escalation into significant drift always alerts, even KS-driven.
    rep = _report([_feature("age", [(0.05, OK, 0.02), (0.05, DRIFT, 0.40)])],
                  labels=("w1", "w2"))
    alerts = detect_alerts(rep)
    assert len(alerts) == 1
    assert alerts[0].direction == "flat"


def test_gap_snapshot_skipped_for_previous_verdict():
    rep = _report([_feature("income", [
        (0.01, OK, 0.02), (None, "n/a", None), (0.50, DRIFT, 0.40)])])
    alerts = detect_alerts(rep)
    assert len(alerts) == 1
    assert alerts[0].previous_verdict == OK
    assert alerts[0].snapshot_index == 2


def test_single_snapshot_never_alerts():
    rep = _report([_feature("age", [(0.50, DRIFT, 0.40)])], labels=("w1",))
    assert detect_alerts(rep) == []


def test_alert_serializes_to_dict():
    rep = _report([_feature("age", [(0.01, OK, 0.02), (0.50, DRIFT, 0.40)])],
                  labels=("w1", "w2"))
    d = detect_alerts(rep)[0].to_dict()
    assert d == {
        "feature": "age", "previous_verdict": OK, "new_verdict": DRIFT,
        "psi": 0.50, "ks": 0.40, "direction": "rising",
        "snapshot_index": 1, "snapshot_label": "w2",
    }


# --- hook specs ---------------------------------------------------------------

def test_parse_alert_hook_valid():
    assert parse_alert_hook("print").kind == "print"
    hook = parse_alert_hook("file:alerts.jsonl")
    assert hook.kind == "file" and hook.path == "alerts.jsonl"


def test_parse_alert_hook_rejects_bad_specs():
    for bad in ("", "file:", "webhook:http://x", "PRINT", "print:extra",
                "stdout", "file"):
        with pytest.raises(ValueError, match="invalid --alert-hook spec"):
            parse_alert_hook(bad)


# --- dispatch -----------------------------------------------------------------

def _alert():
    return Alert(feature="age", previous_verdict=WATCH, new_verdict=DRIFT,
                 psi=0.67, ks=0.33, direction="rising",
                 snapshot_index=2, snapshot_label="w3")


def test_dispatch_file_hook_appends_jsonl(tmp_path):
    path = str(tmp_path / "alerts.jsonl")
    errors = dispatch_alerts([_alert()], [parse_alert_hook(f"file:{path}")])
    assert errors == []
    lines = open(path).read().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["feature"] == "age"
    assert record["new_verdict"] == DRIFT
    assert record["snapshot_label"] == "w3"


def test_dispatch_print_hook(capsys):
    errors = dispatch_alerts([_alert()], [parse_alert_hook("print")])
    assert errors == []
    out = capsys.readouterr().out
    assert "driftwatch alert: age:" in out
    assert WATCH in out and DRIFT in out


def test_dispatch_failing_file_hook_reports_error_not_crash(tmp_path):
    bad = str(tmp_path / "no-such-dir" / "alerts.jsonl")
    errors = dispatch_alerts([_alert()], [parse_alert_hook(f"file:{bad}")])
    assert len(errors) == 1
    assert "file hook failed" in errors[0]


def test_dispatch_one_failing_hook_does_not_block_others(tmp_path, capsys):
    good = str(tmp_path / "alerts.jsonl")
    bad = str(tmp_path / "no-such-dir" / "alerts.jsonl")
    errors = dispatch_alerts(
        [_alert()],
        [parse_alert_hook(f"file:{bad}"), parse_alert_hook(f"file:{good}"),
         parse_alert_hook("print")])
    assert len(errors) == 1
    assert json.loads(open(good).read().strip())["feature"] == "age"
    assert "driftwatch alert: age:" in capsys.readouterr().out


# --- cross-run dedup via state file -------------------------------------------

def test_state_suppresses_repeat_alert(tmp_path):
    state_path = str(tmp_path / "state.json")
    state = load_alert_state(state_path)  # missing file -> fresh state
    assert state == {}
    alerts = [_alert()]
    fresh, suppressed = filter_already_reported(alerts, state)
    assert fresh == alerts and suppressed == 0
    save_alert_state(state_path, record_alerts(state, alerts))
    # Same alert on a re-run of the same snapshots is suppressed.
    reloaded = load_alert_state(state_path)
    fresh, suppressed = filter_already_reported(alerts, reloaded)
    assert fresh == [] and suppressed == 1
    # A fresh escalation on a new snapshot still fires.
    newer = Alert(feature="age", previous_verdict=WATCH, new_verdict=DRIFT,
                  psi=0.70, ks=0.35, direction="rising",
                  snapshot_index=3, snapshot_label="w4")
    fresh, suppressed = filter_already_reported([newer], reloaded)
    assert fresh == [newer] and suppressed == 0


def test_load_alert_state_rejects_corrupt_file(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="cannot read alert state"):
        load_alert_state(str(path))
    path.write_text("[1, 2]")
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_alert_state(str(path))


# --- CLI integration ----------------------------------------------------------

HEADER = ["age", "income", "plan"]


def _make_rows(rng, n, age_shift=0.0):
    rows = []
    for _ in range(n):
        age = max(18, int(rng.gauss(35 + age_shift, 10)))
        income = max(20_000, int(rng.lognormvariate(10.9, 0.5)))
        plan = rng.choices(["basic", "plus", "premium"],
                           weights=(0.5, 0.3, 0.2))[0]
        rows.append([age, income, plan])
    return rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)


def _series(tmp_path):
    ref = tmp_path / "ref.csv"
    _write_csv(ref, _make_rows(random.Random(42), 300))
    snaps = []
    # Shifts chosen so age escalates at the LATEST snapshot only:
    # w1 clean, w2 watch, w3 significant drift -> exactly one alert.
    for seed, shift in [(101, 0.0), (102, 2.0), (103, 10.0)]:
        p = tmp_path / f"snap{shift:g}.csv"
        _write_csv(p, _make_rows(random.Random(seed), 300, age_shift=shift))
        snaps.append(str(p))
    return str(ref), snaps


def test_cli_trend_alert_hook_file_and_json(tmp_path, capsys):
    ref, snaps = _series(tmp_path)
    hook_path = str(tmp_path / "alerts.jsonl")
    json_path = str(tmp_path / "trend.json")
    code = main(["trend", ref, *snaps, "--labels", "w1,w2,w3",
                 "--alert-hook", f"file:{hook_path}",
                 "--alert-hook", "print", "--json", json_path])
    # Exit code unchanged: drifted latest snapshot still gates at 2.
    assert code == EXIT_DRIFT_FOUND
    out = capsys.readouterr().out
    assert "alerts:" in out and "age:" in out
    assert "driftwatch alert: age:" in out

    data = json.loads(open(json_path).read())
    assert len(data["alerts"]) >= 1
    age_alerts = [a for a in data["alerts"] if a["feature"] == "age"]
    assert len(age_alerts) == 1
    assert age_alerts[0]["new_verdict"] == DRIFT
    assert age_alerts[0]["direction"] == "rising"
    assert age_alerts[0]["snapshot_label"] == "w3"
    assert data["hook_errors"] == []

    lines = open(hook_path).read().strip().split("\n")
    assert len(lines) == len(data["alerts"])
    assert json.loads(lines[0])["feature"] == "age"


def test_cli_trend_bad_hook_spec_rejected(tmp_path, capsys):
    ref, snaps = _series(tmp_path)
    code = main(["trend", ref, *snaps, "--alert-hook", "webhook:http://x"])
    assert code == 1
    assert "invalid --alert-hook spec" in capsys.readouterr().err


def test_cli_trend_failing_hook_does_not_crash_report(tmp_path, capsys):
    ref, snaps = _series(tmp_path)
    json_path = str(tmp_path / "trend.json")
    bad = str(tmp_path / "no-such-dir" / "alerts.jsonl")
    code = main(["trend", ref, *snaps, "--alert-hook", f"file:{bad}",
                 "--json", json_path])
    # Trend report still completes; exit code still reflects drift.
    assert code == EXIT_DRIFT_FOUND
    assert "alert hook error" in capsys.readouterr().err
    data = json.loads(open(json_path).read())
    assert len(data["alerts"]) >= 1
    assert len(data["hook_errors"]) == 1


def test_cli_trend_state_file_dedups_across_runs(tmp_path, capsys):
    ref, snaps = _series(tmp_path)
    hook_path = str(tmp_path / "alerts.jsonl")
    state_path = str(tmp_path / "state.json")
    argv = ["trend", ref, *snaps, "--labels", "w1,w2,w3",
            "--alert-hook", f"file:{hook_path}",
            "--alert-state", state_path]
    assert main(argv) == EXIT_DRIFT_FOUND
    first_lines = open(hook_path).read().strip().split("\n")
    assert len(first_lines) >= 1
    # Re-running the same snapshots appends nothing: alerts already reported.
    capsys.readouterr()
    assert main(argv) == EXIT_DRIFT_FOUND
    assert "suppressed" in capsys.readouterr().err
    assert open(hook_path).read().strip().split("\n") == first_lines


def test_cli_trend_stable_series_has_no_alerts(tmp_path):
    ref, _ = _series(tmp_path)
    # Identical snapshots: PSI exactly 0 everywhere, no escalation possible.
    snap = tmp_path / "same.csv"
    _write_csv(snap, _make_rows(random.Random(201), 300))
    json_path = str(tmp_path / "trend.json")
    code = main(["trend", ref, str(snap), str(snap), str(snap),
                 "--json", json_path])
    assert code == 0
    assert json.loads(open(json_path).read())["alerts"] == []
