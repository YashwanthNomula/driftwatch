"""Dataset comparison orchestration: CSV loading, type inference, per-feature drift."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

from . import stats


@dataclass
class FeatureResult:
    name: str
    kind: str  # "numeric" | "categorical" | "dropped" | "new"
    psi: float | None
    ks: float | None
    verdict: str
    n_ref: int
    n_cur: int
    missing_ref: int
    missing_cur: int
    detail: dict = field(default_factory=dict)


@dataclass
class DriftReport:
    features: list[FeatureResult]
    n_ref_rows: int
    n_cur_rows: int
    dropped: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)

    @property
    def n_drifted(self) -> int:
        return sum(1 for f in self.features if f.verdict == "significant drift")

    @property
    def n_watch(self) -> int:
        return sum(1 for f in self.features if f.verdict.startswith("moderate"))


def read_csv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header row found")
        return list(reader.fieldnames), list(reader)


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except (ValueError, TypeError):
        return False


def infer_kind(values: list[str]) -> str:
    """Numeric if every non-missing value parses as float, else categorical."""
    non_missing = [v for v in values if v != "" and v is not None]
    if not non_missing:
        return "categorical"
    return "numeric" if all(_is_float(v) for v in non_missing) else "categorical"


def _split_missing(values: list[str]) -> tuple[list[str], int]:
    present = [v for v in values if v != "" and v is not None]
    return present, len(values) - len(present)


def compare_feature(name: str, ref_vals: list[str], cur_vals: list[str]) -> FeatureResult:
    ref_present, ref_missing = _split_missing(ref_vals)
    cur_present, cur_missing = _split_missing(cur_vals)
    kind = infer_kind(ref_present + cur_present)

    if kind == "numeric":
        ref_nums = [float(v) for v in ref_present]
        cur_nums = [float(v) for v in cur_present]
        psi_value, edges, ref_counts, cur_counts = stats.psi_numeric(ref_nums, cur_nums)
        ks_value = stats.ks_2samp(ref_nums, cur_nums)
        verdict = stats.psi_verdict(psi_value)
        detail = {
            "bin_edges": edges,
            "ref_counts": ref_counts,
            "cur_counts": cur_counts,
            "ref_pct": [c / len(ref_nums) if ref_nums else 0.0 for c in ref_counts],
            "cur_pct": [c / len(cur_nums) if cur_nums else 0.0 for c in cur_counts],
        }
    else:
        psi_value, cats, ref_counts, cur_counts = stats.psi_categorical(ref_present, cur_present)
        ks_value = None
        verdict = stats.psi_verdict(psi_value)
        detail = {
            "categories": cats,
            "ref_counts": ref_counts,
            "cur_counts": cur_counts,
            "ref_pct": [c / len(ref_present) if ref_present else 0.0 for c in ref_counts],
            "cur_pct": [c / len(cur_present) if cur_present else 0.0 for c in cur_counts],
        }

    return FeatureResult(
        name=name, kind=kind, psi=psi_value, ks=ks_value, verdict=verdict,
        n_ref=len(ref_present), n_cur=len(cur_present),
        missing_ref=ref_missing, missing_cur=cur_missing, detail=detail,
    )


def compare_datasets(ref_path: str, cur_path: str,
                      exclude: tuple[str, ...] = ()) -> DriftReport:
    """Compare two CSVs column-by-column; returns a full drift report."""
    ref_cols, ref_rows = read_csv(ref_path)
    cur_cols, cur_rows = read_csv(cur_path)

    ref_set, cur_set = set(ref_cols), set(cur_cols)
    dropped = [c for c in ref_cols if c not in cur_set]
    added = [c for c in cur_cols if c not in ref_set]
    common = [c for c in ref_cols if c in cur_set and c not in exclude]

    features: list[FeatureResult] = []
    for col in common:
        ref_vals = [r.get(col, "") or "" for r in ref_rows]
        cur_vals = [r.get(col, "") or "" for r in cur_rows]
        features.append(compare_feature(col, ref_vals, cur_vals))

    return DriftReport(features=features, n_ref_rows=len(ref_rows),
                       n_cur_rows=len(cur_rows), dropped=dropped, added=added)
