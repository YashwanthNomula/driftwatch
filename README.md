# driftwatch

**Data drift detection in pure Python — zero dependencies.** Compare a reference CSV (your training data)
against a current CSV (your serving data) and find out which features moved, by how much, and whether you
should care. Terminal table, JSON, and a standalone HTML report with per-feature distribution bars.

## Why this matters

Models don't usually break because the code changed — they break because **the world changed**. A signup flow
starts attracting older users, a partner feed starts sending new category codes, a sensor recalibrates. The model
keeps serving with the same confidence while its inputs silently leave the training distribution. Drift detection
is the smoke alarm for production ML: cheap to run on a schedule, and it tells you *which feature* moved before
your metrics do.

## The metrics

- **PSI (Population Stability Index)** — the industry-standard drift score. Distributions are binned (quantile
  bins for numerics, category groups for categoricals) and PSI = Σ (cur% − ref%) · ln(cur% / ref%). Verdicts use
  the standard thresholds: **< 0.1** no significant change, **0.1–0.25** moderate shift (watch), **> 0.25**
  significant drift. Zero-count bins get epsilon smoothing so the score stays finite instead of exploding to `inf`.
- **KS statistic** (numerics only) — the two-sample Kolmogorov–Smirnov distance: the largest gap between the two
  cumulative distributions. A second opinion that doesn't depend on binning choices.

## Quickstart

```bash
python examples/make_examples.py   # generate demo CSVs
python -m driftwatch examples/reference.csv examples/current.csv --target churn
```

## Real demo transcript

The bundled examples simulate a churn model: `current.csv` has its `age` distribution shifted up ~8 years and its
`plan` mix tilted toward premium, while `income` and `region` are untouched. driftwatch catches exactly that:

```
$ python -m driftwatch examples/reference.csv examples/current.csv --target churn
driftwatch: 1200 reference rows vs 1200 current rows
features: 4 compared | significant drift: 1 | watch: 1

feature               type             PSI      KS  verdict
----------------------------------------------------------------------
age                   numeric        0.659   0.333  significant drift
income                numeric        0.012   0.034  no significant change
plan                  categorical    0.193       —  moderate shift — watch
region                categorical    0.005       —  no significant change
```

And the stable pair (same distribution, different seed) stays quiet — no false alarms:

```
$ python -m driftwatch examples/reference_stable.csv examples/current_stable.csv --target churn
driftwatch: 1200 reference rows vs 1200 current rows
features: 4 compared | significant drift: 0 | watch: 0

feature               type             PSI      KS  verdict
----------------------------------------------------------------------
age                   numeric        0.022   0.031  no significant change
income                numeric        0.008   0.023  no significant change
plan                  categorical    0.003       —  no significant change
region                categorical    0.005       —  no significant change
```

Full reports for dashboards and pipelines:

```bash
python -m driftwatch reference.csv current.csv --target churn \
    --report drift_report.html --json drift_report.json
```

Exit codes: `0` = ran clean, no significant drift · `1` = error · `2` = significant drift detected —
so you can wire it straight into CI or a cron job as a deployment gate.

## Edge cases handled

- **Unseen categories** in current data get their own (smoothed) bin — never silently dropped
- **Constant features** collapse to a single bin instead of crashing the quantiler
- **Missing values** are counted and reported per feature, excluded from the distributions
- **Dropped / new columns** between the two CSVs are reported, not compared

## Layout

```
driftwatch/        # library: stats.py (PSI/KS), detect.py (comparison), report.py, cli.py
examples/          # make_examples.py + generated CSV pairs (drifted & stable)
tests/             # 21 pytest tests
```
