"""Alert hooks for drift trends: notify when a feature newly crosses into
drift (or rises into warning), with dedup so an already-drifted feature
does not re-alert on every run.

Dedup rule (two layers):

1. Transition-only: an alert fires only on a verdict *escalation* between
   the previous comparable snapshot and the latest one. A feature that was
   already drifted (or already on watch) in the previous snapshot stays
   silent while it stays there.
2. Optional state file (``--alert-state``): remembers the last reported
   alert per feature (verdict + snapshot label). Re-running the same
   snapshots suppresses the same alert; a fresh escalation on a new
   snapshot still fires.

Hook specs (``--alert-hook``, repeatable):

- ``print`` — print each alert to stdout.
- ``file:PATH`` — append each alert as one JSON object per line (JSONL).

A bad hook spec is rejected with a clear error (exit 1). A hook that fails
at dispatch time (e.g. an unwritable file) is reported to stderr and does
not crash the trend report.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .trend import TrendReport

#: Severity ranking of verdicts. "n/a" (missing snapshot) is never compared.
_SEVERITY = {
    "no significant change": 0,
    "moderate shift — watch": 1,
    "significant drift": 2,
}


@dataclass
class Alert:
    """One newly-escalated feature in the latest trend snapshot."""

    feature: str
    previous_verdict: str
    new_verdict: str
    psi: float | None
    ks: float | None
    direction: str  # rising / falling / flat / n/a (PSI trajectory)
    snapshot_index: int  # 0-based index of the latest snapshot
    snapshot_label: str  # label of the latest snapshot (e.g. "week3")

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "previous_verdict": self.previous_verdict,
            "new_verdict": self.new_verdict,
            "psi": self.psi,
            "ks": self.ks,
            "direction": self.direction,
            "snapshot_index": self.snapshot_index,
            "snapshot_label": self.snapshot_label,
        }

    def summary(self) -> str:
        psi = f"{self.psi:.3f}" if self.psi is not None else "—"
        ks = f"{self.ks:.3f}" if self.ks is not None else "—"
        return (f"{self.feature}: {self.previous_verdict} → {self.new_verdict} "
                f"(PSI {psi}, KS {ks}, {self.direction}, {self.snapshot_label})")


def detect_alerts(report: TrendReport) -> list[Alert]:
    """Find features that escalated into warning/drift at the latest snapshot.

    Only the transition from the previous *comparable* snapshot (skipping
    "n/a" points where the feature was missing) to the latest one counts,
    so a feature that is already drifted does not re-alert. An escalation
    into significant drift always alerts; an escalation into watch alerts
    only when the PSI trajectory is rising — a flat/KS-only blip into watch
    is noise, not a trend.
    """
    alerts: list[Alert] = []
    for f in report.features:
        comparable = [(i, p) for i, p in enumerate(f.points)
                      if p.verdict != "n/a"]
        if len(comparable) < 2:
            continue  # need a previous snapshot to detect a transition
        latest_idx, latest = comparable[-1]
        _prev_idx, prev = comparable[-2]
        if _SEVERITY[latest.verdict] <= _SEVERITY[prev.verdict]:
            continue  # no escalation: stable, recovering, or already there
        if latest.verdict != "significant drift" and f.direction != "rising":
            continue
        alerts.append(Alert(
            feature=f.name,
            previous_verdict=prev.verdict,
            new_verdict=latest.verdict,
            psi=latest.psi,
            ks=latest.ks,
            direction=f.direction,
            snapshot_index=latest_idx,
            snapshot_label=report.labels[latest_idx],
        ))
    return alerts


# ---------------------------------------------------------------------------
# Hook specs
# ---------------------------------------------------------------------------

@dataclass
class AlertHook:
    kind: str  # "print" | "file"
    path: str | None = None


def parse_alert_hook(spec: str) -> AlertHook:
    """Parse one ``--alert-hook`` spec; raise ValueError on a bad spec."""
    if spec == "print":
        return AlertHook(kind="print")
    if spec.startswith("file:"):
        path = spec[len("file:"):]
        if not path:
            raise ValueError(
                f"invalid --alert-hook spec {spec!r}: 'file:' needs a path, "
                "e.g. --alert-hook file:alerts.jsonl")
        return AlertHook(kind="file", path=path)
    raise ValueError(
        f"invalid --alert-hook spec {spec!r}: expected 'print' or 'file:PATH'")


def dispatch_alerts(alerts: list[Alert], hooks: list[AlertHook]) -> list[str]:
    """Deliver ``alerts`` to every hook; return one error string per failure.

    A failing hook never raises: the caller reports the errors and the
    trend report still completes.
    """
    errors: list[str] = []
    for hook in hooks:
        try:
            if hook.kind == "print":
                for alert in alerts:
                    print(f"driftwatch alert: {alert.summary()}")
            elif hook.kind == "file":
                assert hook.path is not None  # parse_alert_hook guarantees it
                with open(hook.path, "a", encoding="utf-8") as fh:
                    for alert in alerts:
                        fh.write(json.dumps(alert.to_dict()) + "\n")
            else:  # pragma: no cover - parse_alert_hook only builds known kinds
                errors.append(f"unknown alert hook kind {hook.kind!r}")
        except OSError as exc:
            errors.append(f"{hook.kind} hook failed: {exc}")
    return errors


# ---------------------------------------------------------------------------
# Cross-run dedup via a state file
# ---------------------------------------------------------------------------

def load_alert_state(path: str) -> dict:
    """Load the per-feature last-reported-alert map; ``{}`` when missing.

    Raises ValueError on a corrupt state file so a broken state file is
    surfaced instead of silently dropping dedup.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read alert state {path!r}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"invalid alert state {path!r}: expected a JSON object")
    return data


def save_alert_state(path: str, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)


def filter_already_reported(alerts: list[Alert], state: dict
                            ) -> tuple[list[Alert], int]:
    """Drop alerts already recorded in ``state``; return (fresh, suppressed)."""
    fresh: list[Alert] = []
    for alert in alerts:
        prev = state.get(alert.feature)
        if (isinstance(prev, dict)
                and prev.get("verdict") == alert.new_verdict
                and prev.get("snapshot") == alert.snapshot_label):
            continue
        fresh.append(alert)
    return fresh, len(alerts) - len(fresh)


def record_alerts(state: dict, alerts: list[Alert]) -> dict:
    """Record ``alerts`` in ``state`` (mutated and returned)."""
    for alert in alerts:
        state[alert.feature] = {
            "verdict": alert.new_verdict,
            "snapshot": alert.snapshot_label,
        }
    return state
