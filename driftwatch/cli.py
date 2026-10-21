"""Command-line interface for driftwatch."""

from __future__ import annotations

import argparse
import sys

from .alerts import (
    Alert,
    AlertHook,
    detect_alerts,
    dispatch_alerts,
    filter_already_reported,
    load_alert_state,
    parse_alert_hook,
    record_alerts,
    save_alert_state,
)
from .config import ThresholdConfig, load_config
from .detect import compare_datasets
from . import report as report_mod
from .trend import trend_series

#: Exit code when at least one feature shows significant drift (useful as a
#: deployment gate in CI/CD or scheduled monitoring).
EXIT_DRIFT_FOUND = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="driftwatch",
        description="Detect data drift between a reference CSV (training) "
                    "and a current CSV (serving), per feature.",
    )
    p.add_argument("reference", help="reference CSV, e.g. training data")
    p.add_argument("current", help="current CSV, e.g. recent serving data")
    p.add_argument("--report", metavar="PATH",
                   help="write a standalone HTML report to PATH")
    p.add_argument("--json", metavar="PATH", dest="json_path",
                   help="write the full JSON report to PATH")
    p.add_argument("--target", metavar="COL",
                   help="column to exclude from comparison (e.g. the label)")
    p.add_argument("--config", metavar="PATH",
                   help="JSON config file with per-feature PSI/KS warning/drift thresholds")
    return p


def warn_unknown_config_features(config: ThresholdConfig,
                                 known: set[str]) -> None:
    """Warn about config features that don't match any compared column."""
    unknown = sorted(n for n in config.features if n not in known)
    if unknown:
        print(f"driftwatch: warning: config lists unknown feature(s): "
              f"{', '.join(unknown)}", file=sys.stderr)


def build_trend_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="driftwatch trend",
        description="Track per-feature drift across successive snapshots: "
                    "compare one reference CSV against each current CSV "
                    "(oldest first) and show the PSI trajectory.",
    )
    p.add_argument("reference", help="reference CSV, e.g. training data")
    p.add_argument("snapshots", nargs="+", metavar="SNAPSHOT",
                   help="current CSVs in chronological order (oldest first)")
    p.add_argument("--labels", metavar="L1,L2,...",
                   help="comma-separated labels for the snapshots "
                        "(default: snapshot file names)")
    p.add_argument("--report", metavar="PATH",
                   help="write a standalone HTML trend report to PATH")
    p.add_argument("--json", metavar="PATH", dest="json_path",
                   help="write the full JSON trend report to PATH")
    p.add_argument("--target", metavar="COL",
                   help="column to exclude from comparison (e.g. the label)")
    p.add_argument("--config", metavar="PATH",
                   help="JSON config file with per-feature PSI/KS warning/drift thresholds")
    p.add_argument("--alert-hook", metavar="SPEC", action="append",
                   dest="alert_hooks", default=None,
                   help="alert sink, repeatable: 'print', 'file:PATH', "
                        "'webhook:URL', or 'email:ADDRESS' "
                        "(alerts fire on verdict escalations at the latest snapshot)")
    p.add_argument("--alert-state", metavar="PATH",
                   help="JSON state file remembering already-reported alerts, "
                        "so re-running the same snapshots does not re-alert")
    return p


def trend_main(argv: list[str]) -> int:
    args = build_trend_parser().parse_args(argv)
    exclude = (args.target,) if args.target else ()
    labels = args.labels.split(",") if args.labels else None
    # Fail fast on a bad hook spec: a typo must be a clear error, never a
    # silent no-op that swallows alerts.
    hooks: list[AlertHook] = []
    for spec in (args.alert_hooks or []):
        try:
            hooks.append(parse_alert_hook(spec))
        except ValueError as exc:
            print(f"driftwatch: error: {exc}", file=sys.stderr)
            return 1
    try:
        config = load_config(args.config) if args.config else None
        result = trend_series(args.reference, args.snapshots,
                              labels=labels, exclude=exclude, config=config)
    except (OSError, ValueError) as exc:
        print(f"driftwatch: error: {exc}", file=sys.stderr)
        return 1

    if config is not None:
        warn_unknown_config_features(
            config, {f.name for f in result.features})

    alerts = detect_alerts(result)
    state = None
    if args.alert_state:
        try:
            state = load_alert_state(args.alert_state)
        except ValueError as exc:
            # A corrupt state file must not kill the trend report: warn and
            # continue without cross-run dedup (transition dedup still applies).
            print(f"driftwatch: warning: {exc} (continuing without dedup)",
                  file=sys.stderr)
            state = None
        else:
            alerts, suppressed = filter_already_reported(alerts, state)
            if suppressed:
                print(f"driftwatch: suppressed {suppressed} "
                      f"previously-reported alert(s)", file=sys.stderr)

    # A failing hook is reported, not fatal: the trend report still completes.
    hook_errors = dispatch_alerts(alerts, hooks)
    for err in hook_errors:
        print(f"driftwatch: alert hook error: {err}", file=sys.stderr)

    if state is not None and not hook_errors:
        # Only remember alerts that were actually delivered; if a hook
        # failed, the next run retries them.
        try:
            save_alert_state(args.alert_state, record_alerts(state, alerts))
        except OSError as exc:
            print(f"driftwatch: warning: could not save alert state: {exc}",
                  file=sys.stderr)

    print(report_mod.render_trend_terminal(result, alerts=alerts))

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            fh.write(report_mod.render_trend_json(
                result, alerts=alerts, hook_errors=hook_errors))
        print(f"\nJSON report written to {args.json_path}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(report_mod.render_trend_html(result))
        print(f"HTML report written to {args.report}")

    # Gate on the latest snapshot: it is the current state of serving data.
    return EXIT_DRIFT_FOUND if result.n_drifted_latest else 0


def main(argv: list[str] | None = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if args_list and args_list[0] == "trend":
        return trend_main(args_list[1:])
    args = build_parser().parse_args(args_list)
    exclude = (args.target,) if args.target else ()
    try:
        config = load_config(args.config) if args.config else None
        result = compare_datasets(args.reference, args.current,
                                  exclude=exclude, config=config)
    except (OSError, ValueError) as exc:
        print(f"driftwatch: error: {exc}", file=sys.stderr)
        return 1

    if config is not None:
        known = {f.name for f in result.features}
        known.update(result.dropped, result.added)
        warn_unknown_config_features(config, known)

    print(report_mod.render_terminal(result))

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            fh.write(report_mod.render_json(result))
        print(f"\nJSON report written to {args.json_path}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(report_mod.render_html(result))
        print(f"HTML report written to {args.report}")

    return EXIT_DRIFT_FOUND if result.n_drifted else 0


if __name__ == "__main__":
    sys.exit(main())
