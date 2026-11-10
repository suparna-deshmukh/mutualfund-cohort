# Mutual Fund Cohort — Activation & Retention Analytics

End-to-end cohort analysis for a mutual-fund investment app: a **6-stage activation funnel**, **90-day retention cohorts** built in SQL, and **RFM segmentation** that surfaces the dormant personas worth a re-engagement push.

**Author:** Suparna Deshmukh
**Stack:** Python · DuckDB (SQL) · pandas · NumPy · matplotlib / seaborn
**Full report:** [mutualfund_report.pdf](mutualfund_report.pdf) · [SQL reference](sql_reference.pdf) · [working spreadsheet](cohort_analysis.numbers)

## The question

Signups overstate a fund app's health. This project traces **50,000 users** through the activation journey — signup → KYC → first investment → SIP → repeat investment → still active at day 90 — and then asks *who* falls out and *which* lapsed investors are worth winning back.

The user and event data is **synthetic** (generated to mirror AMFI-style investment behaviour), which keeps the project fully reproducible and privacy-safe.

## Pipeline

| # | Script | What it does |
|---|--------|--------------|
| 0 | `generate_data.py` | Generates 50,000 users and 82,138 investment events with realistic activation rates, channels, SIP behaviour and churn. Seeded (42) — every run is byte-identical. |
| 1 | `02_funnel.py` | Loads `users.csv` + `events.csv` into DuckDB, runs a 6-stage activation-funnel query, draws the funnel chart, and breaks stage 3 down by source channel. |
| 2 | `03_cohorts.py` | Builds the retention cohort matrix in SQL, exports long + wide forms, renders the cohort heatmap, and prints headline retention metrics. |
| 3 | `04_rfm.py` | Computes Recency / Frequency / Monetary per user in SQL, scores each dimension 1–5 on quintile cuts, maps scores to named segments, and writes `rfm_segments.csv` + the segment chart. |

## Results (from committed `outputs/`)

**Activation funnel** (`outputs/activation_funnel.csv`):

| Stage | Users | Conv. from previous | % of signups |
|-------|------:|--------------------:|-------------:|
| Signed up | 50,000 | — | 100.0 % |
| Completed KYC | 42,567 | 85.1 % | 85.1 % |
| Made first investment | 22,214 | 52.2 % | 44.4 % |
| Set up SIP | 11,041 | 49.7 % | 22.1 % |
| Second investment ≤ 30 days | 11,940 | — | 23.9 % |
| Active at day 90 | 6,894 | 57.7 % | 13.8 % |

The biggest leak is **KYC → first investment** (52 % step conversion) — motivation, not paperwork, is the bottleneck.

**Retention cohorts** (`outputs/cohort_matrix.csv`, `outputs/figures/cohort_heatmap.png`): weekly retention by signup month, weeks 0–12, with cohorts visibly decaying and flattening toward dormancy around week 6.

**RFM segments** (`outputs/rfm_segments.csv`): six personas — **Champions, Loyal, Potential Loyalist, At Risk, Hibernating, Lost** — each with a plain-English description of the recommended re-engagement treatment (printed at the end of `04_rfm.py`).

## Data

| File | What it is |
|------|------------|
| `data/users.csv` | 50,000 synthetic users: signup date, channel, KYC status, activity flags |
| `data/events.csv` | 82,138 synthetic investment events (amounts, dates, SIP vs one-off) |
| `data/mf.duckdb` | DuckDB database with both tables loaded for direct SQL |

## How to run

```bash
# Dependencies (Python 3.9+)
pip install pandas numpy duckdb matplotlib seaborn

python generate_data.py   # only needed to regenerate data/ (seeded — safe to skip)
python 02_funnel.py
python 03_cohorts.py
python 04_rfm.py
```

> **Note:** scripts carry absolute paths in a `PATHS` block at the top — point them at your copy of the folder before running.

## Repository layout

```
mutualfund-cohort/
├── generate_data.py        # seeded synthetic user + event generator
├── 02_funnel.py            # 6-stage activation funnel (SQL + chart)
├── 03_cohorts.py           # retention cohort matrix + heatmap
├── 04_rfm.py               # RFM scoring → named segments
├── data/                   # users.csv, events.csv, mf.duckdb
├── outputs/
│   ├── activation_funnel.csv / funnel_by_channel.csv
│   ├── cohort_long.csv / cohort_matrix.csv
│   ├── rfm_segments.csv
│   └── figures/            # funnel.png, cohort_heatmap.png, rfm_segments.png
├── mutualfund_report.pdf   # full write-up
├── sql_reference.pdf       # every SQL query, with intent
└── cohort_analysis.numbers # working spreadsheet (Numbers/Excel)
```
