"""Generate example CSVs for the driftwatch demo.

Creates two pairs:
  reference.csv / current.csv                 — current.csv has DELIBERATE drift
      * `age` shifted up by ~8 years (numeric drift)
      * `plan` mix shifted toward premium (categorical drift)
      * `income`, `region` unchanged
  reference_stable.csv / current_stable.csv   — same distribution, different
      seed: no drift expected (sanity check)

Run:  python examples/make_examples.py
"""

from __future__ import annotations

import csv
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
N = 1200
HEADER = ["age", "income", "plan", "region", "churn"]


def gen_rows(rng: random.Random, age_shift: float = 0.0,
             plan_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)):
    rows = []
    for _ in range(N):
        age = max(18, int(rng.gauss(35 + age_shift, 10)))
        income = max(20_000, int(rng.lognormvariate(10.9, 0.5)))
        plan = rng.choices(["basic", "plus", "premium"], weights=plan_weights)[0]
        region = rng.choices(["north", "south", "east", "west"])[0]
        churn = "yes" if rng.random() < (0.25 if plan == "basic" else 0.10) else "no"
        rows.append([age, income, plan, region, churn])
    return rows


def write(name: str, rows) -> None:
    path = os.path.join(HERE, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    ref = gen_rows(random.Random(42))
    cur = gen_rows(random.Random(7), age_shift=8.0, plan_weights=(0.3, 0.3, 0.4))
    write("reference.csv", ref)
    write("current.csv", cur)

    ref_s = gen_rows(random.Random(1234))
    cur_s = gen_rows(random.Random(987))
    write("reference_stable.csv", ref_s)
    write("current_stable.csv", cur_s)


if __name__ == "__main__":
    main()
