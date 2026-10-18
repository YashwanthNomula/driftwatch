"""Drift trend over time: compare one reference CSV against a series of
successive current snapshots and report the per-feature drift trajectory.

Designed for scheduled monitoring: run ``driftwatch trend reference.csv
week1.csv week2.csv week3.csv`` after a cron job has dumped each week's
serving sample, and see which features are steadily drifting versus
transiently blipping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .detect import DriftReport, FeatureResult, compare_datasets
from .config import ThresholdConfig, load_config

#: Minimum absolute least-squares slope (PSI per snapshot) to call a
#: trajectory "rising" or "falling" instead of "flat".
SLOPE_TOLERANCE = 0.02

#: PSI value that maps to a full-width bar in the HTML trend report.
HTML_PSI_SCALE = 0.5

_SPARK = "▁▂▃▄▅▆▇"


@dataclass
class TrendPoint:
    """One feature's drift reading at one snapshot."""
    psi: float | None
    verdict: str  # e.g. "significant drift", or "n/a" when not comparable
    ks: float | None = None  # numerics only; None for categoricals / n/a


@dataclass
class TrendFeature:
    name: str
    kind: str
    points: list[TrendPoint]  # one per snapshot, oldest first

    @property
    def psi_series(self) -> list[float | None]:
        return [p.psi for p in self.points]

    @property
    def latest_verdict(self) -> str:
        for p in reversed(self.points):
            if p.verdict != "n/a":
                return p.verdict
        return "n/a"

    @property
    def direction(self) -> str:
        """Trajectory of the PSI series: rising / falling / flat / n/a."""
        return trend_direction(self.psi_series)


@dataclass
class TrendReport:
    features: list[TrendFeature]
    labels: list[str]  # one per snapshot, oldest first
    n_ref_rows: int
    snapshot_rows: list[int]

    @property
    def n_drifted_latest(self) -> int:
        return sum(1 for f in self.features
                   if f.latest_verdict == "significant drift")

    @property
    def n_watch_latest(self) -> int:
        return sum(1 for f in self.features
                   if f.latest_verdict.startswith("moderate"))


def _default_label(path: str) -> str:
    base = os.path.basename(path)
    return base[:-4] if base.lower().endswith(".csv") else base


def trend_series(ref_path: str, snap_paths: list[str],
                 labels: list[str] | None = None,
                 exclude: tuple[str, ...] = (),
                 config: ThresholdConfig | str | None = None) -> TrendReport:
    """Compare ``ref_path`` against each snapshot in order; oldest first.

    ``config`` is an optional :class:`ThresholdConfig` (or path to a JSON
    config file); per-feature thresholds are applied to every snapshot's
    verdicts, with unlisted features falling back to the defaults.
    """
    snaps = list(snap_paths)
    if not snaps:
        raise ValueError("trend needs at least one snapshot CSV")
    if labels is None:
        labels = [_default_label(p) for p in snaps]
    if len(labels) != len(snaps):
        raise ValueError(f"{len(labels)} labels for {len(snaps)} snapshots")
    if isinstance(config, str):
        config = load_config(config)

    reports = [compare_datasets(ref_path, sp, exclude=exclude, config=config)
               for sp in snaps]

    # Feature order follows the first snapshot; any feature that only appears
    # in later snapshots is appended at the end.
    names: list[str] = []
    kinds: dict[str, str] = {}
    for rep in reports:
        for f in rep.features:
            if f.name not in kinds:
                kinds[f.name] = f.kind
                names.append(f.name)
    lookup: list[dict[str, FeatureResult]] = [
        {f.name: f for f in rep.features} for rep in reports
    ]

    features: list[TrendFeature] = []
    for name in names:
        points = []
        for table in lookup:
            f = table.get(name)
            if f is None:
                points.append(TrendPoint(psi=None, verdict="n/a", ks=None))
            else:
                points.append(TrendPoint(psi=f.psi, verdict=f.verdict, ks=f.ks))
        features.append(TrendFeature(name=name, kind=kinds[name], points=points))

    return TrendReport(
        features=features,
        labels=labels,
        n_ref_rows=reports[0].n_ref_rows,
        snapshot_rows=[rep.n_cur_rows for rep in reports],
    )


def trend_direction(psi_values: list[float | None]) -> str:
    """Classify a PSI trajectory via least-squares slope (PSI per snapshot).

    Returns "rising", "falling", "flat", or "n/a" when fewer than two
    comparable points exist.
    """
    xs = [i for i, p in enumerate(psi_values) if p is not None]
    ys = [p for p in psi_values if p is not None]
    if len(xs) < 2:
        return "n/a"
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom if denom else 0.0
    if slope > SLOPE_TOLERANCE:
        return "rising"
    if slope < -SLOPE_TOLERANCE:
        return "falling"
    return "flat"


def sparkline(psi_values: list[float | None]) -> str:
    """Miniature trajectory bar: one glyph per snapshot, scaled to the
    feature's own PSI range. Missing snapshots render as a gap."""
    present = [v for v in psi_values if v is not None]
    if not present:
        return " " * len(psi_values)
    lo, hi = min(present), max(present)
    span = hi - lo
    out = []
    for v in psi_values:
        if v is None:
            out.append(" ")
        else:
            idx = 0 if span == 0 else round((v - lo) / span * (len(_SPARK) - 1))
            out.append(_SPARK[idx])
    return "".join(out)


_DIRECTION_ARROW = {"rising": "↑", "falling": "↓", "flat": "→", "n/a": "·"}
