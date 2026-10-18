"""Tests for per-feature threshold configs: parsing, validation, and application."""

import json
import os

import pytest

from driftwatch import stats
from driftwatch.cli import EXIT_DRIFT_FOUND, main
from driftwatch.config import (ConfigError, ThresholdConfig, ThresholdSet,
                               load_config)
from driftwatch.detect import compare_datasets, compare_feature
from driftwatch.trend import trend_series

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "examples")
REF = os.path.join(EXAMPLES, "reference.csv")
CUR = os.path.join(EXAMPLES, "current.csv")
REF_S = os.path.join(EXAMPLES, "reference_stable.csv")
CUR_S = os.path.join(EXAMPLES, "current_stable.csv")


def write_config(tmp_path, data, name="driftwatch.json"):
    path = str(tmp_path / name)
    with open(path, "w") as fh:
        json.dump(data, fh)
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_load_config_defaults_and_overrides(tmp_path):
    path = write_config(tmp_path, {
        "defaults": {"psi_warning": 0.05, "psi_drift": 0.2},
        "features": {"age": {"psi_drift": 0.4}, "income": {}},
    })
    cfg = load_config(path)
    assert cfg.defaults.psi_warning == 0.05
    assert cfg.defaults.psi_drift == 0.2
    # Unspecified keys fall back to the (possibly overridden) defaults.
    assert cfg.defaults.ks_warning == stats.KS_NO_CHANGE
    age = cfg.thresholds_for("age")
    assert age.psi_drift == 0.4
    assert age.psi_warning == 0.05  # inherits config defaults
    assert age.ks_drift == stats.KS_MODERATE
    # Empty per-feature block inherits everything.
    assert cfg.thresholds_for("income") == cfg.defaults


def test_unlisted_feature_falls_back_to_builtin_defaults(tmp_path):
    path = write_config(tmp_path, {"features": {"age": {"psi_drift": 0.9}}})
    cfg = load_config(path)
    assert cfg.defaults == ThresholdSet()  # builtins untouched
    other = cfg.thresholds_for("some_other_feature")
    assert other.psi_warning == stats.PSI_NO_CHANGE
    assert other.psi_drift == stats.PSI_MODERATE


def test_empty_config_uses_builtin_defaults(tmp_path):
    cfg = load_config(write_config(tmp_path, {}))
    assert cfg.defaults == ThresholdSet()
    assert cfg.features == {}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_reject_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nope/no-such-config.json")


def test_reject_malformed_json(tmp_path):
    path = str(tmp_path / "bad.json")
    open(path, "w").write("{not valid json")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_config(path)


def test_reject_non_object_config(tmp_path):
    with pytest.raises(ConfigError, match="must be a JSON object"):
        load_config(write_config(tmp_path, [1, 2, 3]))


def test_reject_unknown_top_level_key(tmp_path):
    with pytest.raises(ConfigError, match="unknown top-level"):
        load_config(write_config(tmp_path, {"thresholds": {}}))


def test_reject_unknown_threshold_key(tmp_path):
    with pytest.raises(ConfigError, match="unknown threshold key"):
        load_config(write_config(tmp_path,
                                 {"features": {"age": {"psi_alarm": 0.5}}}))


def test_reject_negative_threshold(tmp_path):
    with pytest.raises(ConfigError, match="must be >= 0"):
        load_config(write_config(tmp_path,
                                 {"defaults": {"psi_warning": -0.1}}))


def test_reject_non_numeric_threshold(tmp_path):
    with pytest.raises(ConfigError, match="must be a number"):
        load_config(write_config(tmp_path,
                                 {"features": {"age": {"ks_drift": "high"}}}))


def test_reject_bool_threshold(tmp_path):
    with pytest.raises(ConfigError, match="must be a number"):
        load_config(write_config(tmp_path,
                                 {"defaults": {"psi_drift": True}}))


def test_reject_non_monotonic_thresholds(tmp_path):
    with pytest.raises(ConfigError, match="must be below"):
        load_config(write_config(tmp_path, {
            "features": {"age": {"psi_warning": 0.3, "psi_drift": 0.2}}}))


def test_reject_equal_warning_and_drift(tmp_path):
    with pytest.raises(ConfigError, match="must be below"):
        load_config(write_config(tmp_path,
                                 {"defaults": {"ks_warning": 0.2,
                                               "ks_drift": 0.2}}))


def test_reject_bad_feature_name(tmp_path):
    with pytest.raises(ConfigError, match="non-empty string"):
        load_config(write_config(tmp_path, {"features": {"": {"psi_drift": 1}}}))


# ---------------------------------------------------------------------------
# Application to detect output
# ---------------------------------------------------------------------------

def test_per_feature_thresholds_applied_to_detect(tmp_path):
    # age drifts badly (psi ~0.66); a lenient per-feature config downgrades it.
    path = write_config(tmp_path,
                        {"features": {"age": {"psi_warning": 0.5,
                                              "psi_drift": 0.8,
                                              "ks_warning": 0.4,
                                              "ks_drift": 0.5}}})
    rep = compare_datasets(REF, CUR, config=load_config(path))
    by_name = {f.name: f for f in rep.features}
    assert by_name["age"].verdict == "moderate shift — watch"
    # Unlisted features keep the default verdicts.
    assert by_name["income"].verdict == "no significant change"
    assert by_name["plan"].verdict == "moderate shift — watch"


def test_strict_config_raises_verdict_and_is_fatal(tmp_path):
    # income is stable (psi ~0.012); a strict config escalates it to drift.
    path = write_config(tmp_path,
                        {"features": {"income": {"psi_warning": 0.001,
                                                 "psi_drift": 0.005}}})
    rep = compare_datasets(REF_S, CUR_S, config=load_config(path))
    by_name = {f.name: f for f in rep.features}
    assert by_name["income"].verdict == "significant drift"
    assert rep.n_drifted >= 1


def test_compare_datasets_accepts_config_path_string(tmp_path):
    path = write_config(tmp_path,
                        {"features": {"age": {"psi_warning": 0.5,
                                              "psi_drift": 0.8,
                                              "ks_warning": 0.4,
                                              "ks_drift": 0.5}}})
    rep = compare_datasets(REF, CUR, config=path)
    by_name = {f.name: f for f in rep.features}
    assert by_name["age"].verdict == "moderate shift — watch"


def test_ks_thresholds_feed_the_verdict():
    # A tiny uniform shift barely moves bin proportions (PSI ~ 0) but KS sees
    # the CDF gap; with a strict KS cutoff the feature is flagged by KS alone.
    ref = [i / 1000 for i in range(1000)]
    cur = [v + 0.01 for v in ref]
    th = ThresholdSet(ks_warning=0.001, ks_drift=0.005)
    f = compare_feature("x", [str(v) for v in ref], [str(v) for v in cur],
                        thresholds=th)
    assert f.psi < 0.01
    assert f.ks > 0.005
    assert f.verdict == "significant drift"


def test_ks_verdict_helpers():
    assert stats.ks_verdict(0.05) == "no significant change"
    assert stats.ks_verdict(0.1) == "moderate shift — watch"
    assert stats.ks_verdict(0.3) == "moderate shift — watch"
    assert stats.ks_verdict(0.31) == "significant drift"
    assert stats.ks_verdict(0.5, warning=0.6, drift=0.8) == "no significant change"
    assert stats.worst_verdict("no significant change",
                               "significant drift") == "significant drift"


# ---------------------------------------------------------------------------
# Trend + CLI wiring
# ---------------------------------------------------------------------------

def test_trend_series_applies_config(tmp_path):
    path = write_config(tmp_path,
                        {"features": {"age": {"psi_warning": 0.5,
                                              "psi_drift": 0.8,
                                              "ks_warning": 0.4,
                                              "ks_drift": 0.5}}})
    rep = trend_series(REF, [CUR_S, CUR], config=load_config(path))
    age = next(f for f in rep.features if f.name == "age")
    # Latest snapshot (drifted) verdict now follows the lenient thresholds.
    assert age.latest_verdict == "moderate shift — watch"


def test_cli_config_changes_exit_code(tmp_path, capsys):
    # Stable pair exits 0 by default; a strict config flips income to drift.
    path = write_config(tmp_path,
                        {"features": {"income": {"psi_warning": 0.001,
                                                 "psi_drift": 0.005}}})
    code = main([REF_S, CUR_S, "--target", "churn", "--config", path])
    assert code == EXIT_DRIFT_FOUND


def test_cli_invalid_config_exits_1(tmp_path, capsys):
    path = write_config(tmp_path, {"defaults": {"psi_drift": -1}})
    code = main([REF, CUR, "--config", path])
    assert code == 1
    assert "error" in capsys.readouterr().err


def test_cli_warns_on_unknown_config_feature(tmp_path, capsys):
    path = write_config(tmp_path,
                        {"features": {"not_a_column": {"psi_drift": 0.9}}})
    code = main([REF_S, CUR_S, "--target", "churn", "--config", path])
    assert code == 0
    assert "unknown feature" in capsys.readouterr().err
