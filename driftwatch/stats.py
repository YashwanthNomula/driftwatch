"""Core drift statistics: PSI and the two-sample KS statistic. Pure Python, zero dependencies."""

from __future__ import annotations

import bisect
import math

#: Standard PSI interpretation thresholds.
PSI_NO_CHANGE = 0.1
PSI_MODERATE = 0.25

#: Smoothing applied to zero-probability bins so PSI stays finite.
EPSILON = 1e-4


def quantile(sorted_vals: list[float], q: float) -> float | None:
    """Linear-interpolation quantile of an already-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def quantile_bin_edges(ref_values: list[float], n_bins: int = 10) -> list[float]:
    """Bin edges from reference-data quantiles; duplicate edges are dropped.

    A constant feature collapses to a single bin (edges ``[-inf, +inf]``).
    """
    vals = sorted(ref_values)
    if not vals:
        return [float("-inf"), float("inf")]
    edges = [float("-inf")]
    for i in range(1, n_bins):
        q = quantile(vals, i / n_bins)
        if q is not None and q > edges[-1]:
            edges.append(q)
    edges.append(float("inf"))
    return edges


def assign_bins(values: list[float], edges: list[float]) -> list[int]:
    """Assign each value to a bin index given ascending edges starting at -inf."""
    # bisect on interior edges; edges[0] is -inf and edges[-1] is +inf
    inner = edges[1:-1]
    return [bisect.bisect_right(inner, v) for v in values]


def binned_counts(values: list[float], edges: list[float]) -> list[int]:
    n = len(edges) - 1
    counts = [0] * n
    for b in assign_bins(values, edges):
        counts[b] += 1
    return counts


def psi_from_counts(ref_counts: list[int], cur_counts: list[int],
                    epsilon: float = EPSILON) -> float:
    """Population Stability Index from per-bin counts.

    Zero-probability bins are smoothed with ``epsilon`` so the index stays
    finite instead of exploding to ``inf``.
    """
    n_ref = sum(ref_counts)
    n_cur = sum(cur_counts)
    total = 0.0
    for r, c in zip(ref_counts, cur_counts):
        rp = (r / n_ref) if n_ref else 0.0
        cp = (c / n_cur) if n_cur else 0.0
        rp = rp if rp > 0 else epsilon
        cp = cp if cp > 0 else epsilon
        total += (cp - rp) * math.log(cp / rp)
    return total


def psi_numeric(ref_values: list[float], cur_values: list[float],
                n_bins: int = 10) -> tuple[float, list[float], list[int], list[int]]:
    """PSI for a numeric feature; bins are learned from the reference data.

    Returns ``(psi, edges, ref_counts, cur_counts)``.
    """
    edges = quantile_bin_edges(ref_values, n_bins)
    ref_counts = binned_counts(ref_values, edges)
    cur_counts = binned_counts(cur_values, edges)
    return psi_from_counts(ref_counts, cur_counts), edges, ref_counts, cur_counts


def psi_categorical(ref_values: list[str], cur_values: list[str]
                    ) -> tuple[float, list[str], list[int], list[int]]:
    """PSI for a categorical feature over the union of observed categories.

    Categories seen only in the current data (unseen at training time) are
    kept as their own bin — smoothed, never dropped.
    """
    cats: list[str] = []
    seen: set[str] = set()
    for v in list(ref_values) + list(cur_values):
        if v not in seen:
            seen.add(v)
            cats.append(v)
    ref_counts = [0] * len(cats)
    cur_counts = [0] * len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    for v in ref_values:
        ref_counts[idx[v]] += 1
    for v in cur_values:
        cur_counts[idx[v]] += 1
    return psi_from_counts(ref_counts, cur_counts), cats, ref_counts, cur_counts


def ks_2samp(a: list[float], b: list[float]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic: max |CDF_a - CDF_b|."""
    if not a or not b:
        return 0.0
    sa = sorted(a)
    sb = sorted(b)
    na, nb = len(sa), len(sb)
    i = j = 0
    best = 0.0
    # Sweep the combined sorted grid; evaluate the CDF gap at each step.
    points = sorted(set(sa) | set(sb))
    for x in points:
        while i < na and sa[i] <= x:
            i += 1
        while j < nb and sb[j] <= x:
            j += 1
        gap = abs(i / na - j / nb)
        if gap > best:
            best = gap
    return best


def psi_verdict(psi_value: float) -> str:
    """Human verdict for a PSI value using standard thresholds."""
    if psi_value < PSI_NO_CHANGE:
        return "no significant change"
    if psi_value <= PSI_MODERATE:
        return "moderate shift — watch"
    return "significant drift"
