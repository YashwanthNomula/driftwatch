"""Report rendering: terminal table, JSON, and a standalone HTML report."""

from __future__ import annotations

import html
import json

from .detect import DriftReport, FeatureResult


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def render_terminal(report: DriftReport) -> str:
    lines = [
        f"driftwatch: {report.n_ref_rows} reference rows vs {report.n_cur_rows} current rows",
        f"features: {len(report.features)} compared | "
        f"significant drift: {report.n_drifted} | watch: {report.n_watch}",
        "",
        f"{'feature':<22}{'type':<12}{'PSI':>8}{'KS':>8}  verdict",
        "-" * 70,
    ]
    for f in report.features:
        lines.append(f"{f.name:<22}{f.kind:<12}{_fmt(f.psi):>8}{_fmt(f.ks):>8}  {f.verdict}")
    if report.dropped:
        lines.append("")
        lines.append("dropped columns (in reference only): " + ", ".join(report.dropped))
    if report.added:
        lines.append("new columns (in current only): " + ", ".join(report.added))
    return "\n".join(lines)


def to_dict(report: DriftReport) -> dict:
    def feat(f: FeatureResult) -> dict:
        return {
            "name": f.name, "kind": f.kind, "psi": f.psi, "ks": f.ks,
            "verdict": f.verdict, "n_ref": f.n_ref, "n_cur": f.n_cur,
            "missing_ref": f.missing_ref, "missing_cur": f.missing_cur,
            "detail": f.detail,
        }
    return {
        "n_ref_rows": report.n_ref_rows,
        "n_cur_rows": report.n_cur_rows,
        "n_drifted": report.n_drifted,
        "n_watch": report.n_watch,
        "dropped_columns": report.dropped,
        "new_columns": report.added,
        "features": [feat(f) for f in report.features],
    }


def render_json(report: DriftReport) -> str:
    return json.dumps(to_dict(report), indent=2, default=str)


def _bar(pct: float, color: str) -> str:
    width = max(1.0, min(100.0, pct * 100))
    return (f'<div class="bar-track"><div class="bar" style="width:{width:.1f}%;'
            f'background:{color}"></div></div>')


def _feature_card(f: FeatureResult) -> str:
    name = html.escape(f.name)
    if f.kind == "numeric":
        edges = f.detail["bin_edges"]
        labels = []
        for i in range(len(edges) - 1):
            lo = "−∞" if edges[i] == float("-inf") else f"{edges[i]:.2f}"
            hi = "+∞" if edges[i + 1] == float("inf") else f"{edges[i + 1]:.2f}"
            labels.append(f"{lo} – {hi}")
    else:
        labels = [html.escape(str(c)) for c in f.detail["categories"]]
    rows = []
    for lab, rp, cp in zip(labels, f.detail["ref_pct"], f.detail["cur_pct"]):
        rows.append(
            "<tr>"
            f"<td class='lab'>{lab}</td>"
            f"<td>{_bar(rp, '#3b82f6')}<span class='pct'>{rp * 100:.1f}%</span></td>"
            f"<td>{_bar(cp, '#f59e0b')}<span class='pct'>{cp * 100:.1f}%</span></td>"
            "</tr>"
        )
    ks_line = f"KS = {f.ks:.3f} · " if f.ks is not None else ""
    verdict_class = ("drift" if f.verdict == "significant drift"
                     else "watch" if f.verdict.startswith("moderate") else "ok")
    return (
        f"<section class='card'>"
        f"<h3>{name} <span class='kind'>{f.kind}</span> "
        f"<span class='verdict {verdict_class}'>{html.escape(f.verdict)}</span></h3>"
        f"<p class='meta'>PSI = {f.psi:.3f} · {ks_line}"
        f"n_ref = {f.n_ref}, n_cur = {f.n_cur}"
        f"{f', missing: ref {f.missing_ref} / cur {f.missing_cur}' if (f.missing_ref or f.missing_cur) else ''}</p>"
        "<table><tr><th>bin</th><th>reference</th><th>current</th></tr>"
        + "".join(rows) + "</table></section>"
    )


def render_html(report: DriftReport) -> str:
    cards = "\n".join(_feature_card(f) for f in report.features)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>driftwatch report</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1f2937; }}
h1 {{ font-size: 1.5rem; }} .summary {{ color: #4b5563; }}
.card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem 1.25rem; margin: 1.25rem 0; }}
.card h3 {{ margin: 0 0 .25rem; font-size: 1.05rem; }}
.kind {{ font-size: .75rem; color: #6b7280; font-weight: normal; }}
.meta {{ color: #4b5563; font-size: .85rem; }}
.verdict {{ font-size: .75rem; padding: .15rem .5rem; border-radius: 999px; }}
.verdict.ok {{ background: #dcfce7; color: #166534; }}
.verdict.watch {{ background: #fef9c3; color: #854d0e; }}
.verdict.drift {{ background: #fee2e2; color: #991b1b; }}
table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
th {{ text-align: left; color: #6b7280; font-weight: 600; padding: .25rem .5rem; }}
td {{ padding: .25rem .5rem; border-top: 1px solid #f3f4f6; vertical-align: middle; }}
td.lab {{ white-space: nowrap; max-width: 180px; overflow: hidden; text-overflow: ellipsis; }}
.bar-track {{ display: inline-block; width: 55%; background: #f3f4f6; border-radius: 4px; vertical-align: middle; }}
.bar {{ height: 12px; border-radius: 4px; }}
.pct {{ margin-left: .5rem; color: #4b5563; }}
.legend span {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: .3rem; vertical-align: baseline; }}
</style></head><body>
<h1>driftwatch report</h1>
<p class="summary">{report.n_ref_rows} reference rows vs {report.n_cur_rows} current rows ·
{len(report.features)} features compared ·
<strong>{report.n_drifted}</strong> with significant drift ·
<strong>{report.n_watch}</strong> to watch</p>
<p class="legend"><span style="background:#3b82f6"></span>reference
<span style="background:#f59e0b;margin-left:1rem"></span>current</p>
{cards}
</body></html>
"""
