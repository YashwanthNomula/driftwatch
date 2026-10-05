"""Command-line interface for driftwatch."""

from __future__ import annotations

import argparse
import sys

from .detect import compare_datasets
from . import report as report_mod

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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exclude = (args.target,) if args.target else ()
    try:
        result = compare_datasets(args.reference, args.current, exclude=exclude)
    except (OSError, ValueError) as exc:
        print(f"driftwatch: error: {exc}", file=sys.stderr)
        return 1

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
