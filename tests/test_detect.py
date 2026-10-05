"""Tests for driftwatch.detect: CSV loading, type inference, missing values."""

import os

from driftwatch.detect import compare_datasets, compare_feature, infer_kind

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "examples")


def test_infer_kind():
    assert infer_kind(["1", "2.5", "3"]) == "numeric"
    assert infer_kind(["a", "b", "c"]) == "categorical"
    assert infer_kind(["1", "two", "3"]) == "categorical"
    assert infer_kind([]) == "categorical"
    assert infer_kind(["", ""]) == "categorical"


def test_compare_feature_missing_values_counted():
    f = compare_feature("x", ["1", "", "3", ""], ["1", "2", "3", "4"])
    assert f.missing_ref == 2
    assert f.missing_cur == 0
    assert f.n_ref == 2 and f.n_cur == 4


def test_constant_numeric_feature():
    f = compare_feature("c", ["7"] * 50, ["7"] * 50)
    assert f.kind == "numeric"
    assert f.psi == 0.0
    assert f.verdict == "no significant change"


def test_dropped_and_added_columns_reported():
    import csv
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rp, cp = os.path.join(d, "r.csv"), os.path.join(d, "c.csv")
        with open(rp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["a", "gone"])
            w.writerows([["1", "x"], ["2", "y"]])
        with open(cp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["a", "newbie"])
            w.writerows([["1", "p"], ["2", "q"]])
        rep = compare_datasets(rp, cp)
        assert rep.dropped == ["gone"]
        assert rep.added == ["newbie"]
        assert [f.name for f in rep.features] == ["a"]


def test_target_exclusion():
    rep = compare_datasets(os.path.join(EXAMPLES, "reference.csv"),
                           os.path.join(EXAMPLES, "current.csv"),
                           exclude=("churn",))
    assert "churn" not in [f.name for f in rep.features]


def test_end_to_end_drifted_examples():
    rep = compare_datasets(os.path.join(EXAMPLES, "reference.csv"),
                           os.path.join(EXAMPLES, "current.csv"))
    by_name = {f.name: f for f in rep.features}
    assert by_name["age"].verdict == "significant drift"
    assert by_name["age"].psi > 0.25
    assert by_name["income"].verdict == "no significant change"
    assert rep.n_drifted >= 1


def test_end_to_end_stable_examples():
    rep = compare_datasets(os.path.join(EXAMPLES, "reference_stable.csv"),
                           os.path.join(EXAMPLES, "current_stable.csv"))
    assert rep.n_drifted == 0
    assert rep.n_watch == 0
