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
  cumulative distributions. A second opinion that doesn't depend on binning choices. Default cutoffs: **< 0.1** no
  significant change, **0.1–0.3** moderate shift, **> 0.3** significant drift.
- Each feature's verdict is the **worse** of its PSI and KS verdicts (KS applies to numerics only).

All four cutoffs (PSI warning/drift, KS warning/drift) are configurable globally and per feature — see
"Per-feature thresholds" below.

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

## Trend mode: drift over time

One comparison tells you *what* moved; a trend tells you *how fast it's getting worse*.
Dump each week's serving sample to a CSV and compare the whole series against the reference:

```bash
python -m driftwatch trend reference.csv week1.csv week2.csv week3.csv \
    --labels "week1,week2,week3"
```

Each feature gets a PSI trajectory — values per snapshot, a sparkline, and a
rising/falling/flat direction from the least-squares slope. The exit code gates on the
latest snapshot, and `--json` / `--report` produce the full trend timeline for dashboards:

```
$ python -m driftwatch trend reference.csv week1.csv week2.csv week3.csv --labels "week1,week2,week3"
driftwatch trend: 1200 reference rows vs 3 snapshots
features tracked: 4 | significant drift (latest): 1 | watch (latest): 1

feature               type           week1   week2   week3  trend     verdict
----------------------------------------------------------------------------------
age                   numeric        0.011   0.189   0.670  ▁▃▇↑ rising significant drift
income                numeric        0.012   0.010   0.015  ▃▁▇→ flat   no significant change
plan                  categorical    0.004   0.012   0.212  ▁▁▇↑ rising moderate shift — watch
region                categorical    0.004   0.001   0.008  ▃▁▇→ flat   no significant change

latest snapshot (week3): 1200 rows
```

`age` has been drifting for three weeks and crossed the significant-drift threshold;
`plan` only started moving in week 3 — worth watching, not yet alarming.
Missing features in a snapshot are marked `n/a` instead of breaking the series.

### Alert hooks: notify on new escalations

`--alert-hook` fires when a feature **newly escalates** at the latest snapshot —
into significant drift, or rising into watch. A feature that was already
drifted (or already on watch) in the previous snapshot stays silent, so a
scheduled run doesn't spam you about the same drift every week.

```bash
python -m driftwatch trend reference.csv week1.csv week2.csv week3.csv \
    --alert-hook print --alert-hook file:alerts.jsonl \
    --alert-state alert_state.json --json trend.json
```

```
alerts: 1 new
  ! age: moderate shift — watch → significant drift (PSI 0.670, KS 0.333, rising, week3)
```

- Sinks (repeatable): `print` writes `driftwatch alert: ...` lines to stdout;
  `file:PATH` appends one JSON object per alert (JSONL) — same records that
  land in the trend JSON's `alerts` array; `webhook:URL` POSTs all alerts as
  one JSON payload `{"source": "driftwatch", "alerts": [...]}` to the URL;
  `email:ADDRESS` sends one email per alert via SMTP.
- Email needs `DRIFTWATCH_SMTP_HOST` + `DRIFTWATCH_SMTP_FROM` in the
  environment (plus `DRIFTWATCH_SMTP_USER` / `DRIFTWATCH_SMTP_PASS` if your
  server needs auth, `DRIFTWATCH_SMTP_PORT` to override 587, and
  `DRIFTWATCH_SMTP_SSL=1` for implicit TLS). Passwords are never written to
  a config file — same pattern as sitepulse's notifiers.
- `--alert-state PATH` remembers already-reported alerts (feature → verdict +
  snapshot label), so re-running the same snapshots suppresses duplicates;
  a fresh escalation on a new snapshot still fires.
- Each alert record carries `feature`, `previous_verdict`, `new_verdict`,
  `psi`, `ks`, `direction`, `snapshot_index`, and `snapshot_label`.
- Fail-closed: a bad hook spec (anything but `print` / `file:PATH` /
  `webhook:URL` / `email:ADDRESS`) is
  rejected with a clear error and exit 1; a hook that fails at dispatch
  (e.g. an unwritable file, an unreachable webhook, missing SMTP config) is
  reported on stderr and in the JSON
  `hook_errors` array without crashing the report. Alerts never change the
  exit code — the gate still keys off the latest snapshot's drift.

## Per-feature thresholds via config file

Not every feature deserves the same alarm level: a stable-but-critical feature should page earlier, a naturally
noisy one later. A JSON config file (pure JSON — no new dependencies) overrides the default cutoffs globally and
per feature:

```json
{
  "defaults": {"psi_warning": 0.1, "psi_drift": 0.25,
               "ks_warning": 0.1, "ks_drift": 0.3},
  "features": {
    "age":    {"psi_warning": 0.05, "psi_drift": 0.15},
    "income": {"ks_drift": 0.4}
  }
}
```

```bash
python -m driftwatch reference.csv current.csv --config driftwatch.json
python -m driftwatch trend reference.csv week1.csv week2.csv --config driftwatch.json
```

Rules:

- Any key you leave out falls back to the global default; any feature you don't list uses the defaults entirely
  (the shown `defaults` block is optional — omit it to keep the built-ins).
- Both PSI and KS thresholds participate in each feature's verdict, which stays the worse of the two opinions.
- Invalid configs are rejected with a clear error: bad JSON, unknown keys, non-numeric or negative thresholds,
  and `warning >= drift` cutoffs. A config naming a feature that isn't in the data prints a warning, not an error.

## Edge cases handled

- **Unseen categories** in current data get their own (smoothed) bin — never silently dropped
- **Constant features** collapse to a single bin instead of crashing the quantiler
- **Missing values** are counted and reported per feature, excluded from the distributions
- **Dropped / new columns** between the two CSVs are reported, not compared

## Layout

```
driftwatch/        # library: stats.py (PSI/KS), detect.py (comparison), trend.py (trend over time), alerts.py (alert hooks), notify.py (webhook/email delivery), report.py, cli.py, config.py (per-feature thresholds)
examples/          # make_examples.py + generated CSV pairs (drifted & stable)
tests/             # 83 pytest tests
```
