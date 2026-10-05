"""CLI end-to-end tests: exit codes, JSON/HTML outputs."""

import json
import os

from driftwatch.cli import EXIT_DRIFT_FOUND, main

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "examples")
REF = os.path.join(EXAMPLES, "reference.csv")
CUR = os.path.join(EXAMPLES, "current.csv")
REF_S = os.path.join(EXAMPLES, "reference_stable.csv")
CUR_S = os.path.join(EXAMPLES, "current_stable.csv")


def test_cli_drifted_exits_with_drift_code(capsys):
    code = main([REF, CUR, "--target", "churn"])
    assert code == EXIT_DRIFT_FOUND
    out = capsys.readouterr().out
    assert "significant drift" in out
    assert "age" in out


def test_cli_stable_exits_zero(capsys):
    code = main([REF_S, CUR_S, "--target", "churn"])
    assert code == 0
    out = capsys.readouterr().out
    assert "significant drift: 0" in out


def test_cli_writes_json_and_html(tmp_path, capsys):
    jp = str(tmp_path / "out.json")
    hp = str(tmp_path / "report.html")
    code = main([REF, CUR, "--target", "churn", "--json", jp, "--report", hp])
    assert code == EXIT_DRIFT_FOUND
    data = json.loads(open(jp).read())
    assert data["n_drifted"] >= 1
    assert any(f["name"] == "age" and f["verdict"] == "significant drift"
               for f in data["features"])
    html_text = open(hp).read()
    assert "<html" in html_text and "age" in html_text


def test_cli_missing_file_errors(capsys):
    code = main(["/nope/missing.csv", CUR])
    assert code == 1
    assert "error" in capsys.readouterr().err
