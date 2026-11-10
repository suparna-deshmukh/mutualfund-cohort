"""
03_cohorts.py
Weekly retention cohort analysis for the mutual-fund cohort project.

Steps:
  1. Build cohort matrix in SQL  → cohort_long.csv
  2. Pivot to wide matrix        → cohort_matrix.csv
  3. Seaborn heatmap             → figures/cohort_heatmap.png
  4. Print key retention metrics

Author : Suparna Deshmukh
Project: mutualfund-cohort
"""

# ─────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────
import os
import duckdb
import pandas as pd
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────
BASE_DIR      = "/Users/suparna_deshmukh/projects/mutualfund-cohort"
DB_PATH       = os.path.join(BASE_DIR, "data/mf.duckdb")

OUT_DIR       = os.path.join(BASE_DIR, "outputs")
FIG_DIR       = os.path.join(OUT_DIR,  "figures")
LONG_CSV      = os.path.join(OUT_DIR,  "cohort_long.csv")
MATRIX_CSV    = os.path.join(OUT_DIR,  "cohort_matrix.csv")
HEATMAP_PNG   = os.path.join(FIG_DIR,  "cohort_heatmap.png")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MAX_WEEK      = 12     # track weeks 0 → 12  (covers 0–90 days)

# ─────────────────────────────────────────────────────────
# OPEN DATABASE  (read_only so we never accidentally mutate)
# ─────────────────────────────────────────────────────────
con = duckdb.connect(DB_PATH, read_only=True)
print()
print("=" * 60)
print("  03_cohorts.py — Retention Cohort Analysis")
print("=" * 60)

# ─────────────────────────────────────────────────────────
# STEP 1 — BUILD COHORT MATRIX WITH SQL CTEs
# ─────────────────────────────────────────────────────────
print()
print("▶  STEP 1 — Building cohort retention matrix in SQL …")

# ── What counts as an "active" event? ────────────────────
# Any investment action: first_investment, lumpsum, sip_setup,
# sip_installment.  We exclude 'redemption' because withdrawing
# money is not a signal of engagement — it's the opposite.
ACTIVE_TYPES = "('first_investment','lumpsum','sip_setup','sip_installment')"

# ── How the SQL is structured (plain-English walkthrough) ─
#
# CTE 1 — user_cohorts
#   Attach a signup_month label (e.g. "2025-08") to every user.
#
# CTE 2 — weeks_spine
#   Generate the integers 0, 1, 2 … MAX_WEEK using
#   DuckDB's generate_series() function.
#   This is the "week skeleton" we'll use to check every week
#   for every user.
#
# CTE 3 — user_weeks
#   CROSS JOIN user_cohorts × weeks_spine.
#   This creates one row per (user, week) combination.
#   50 K users × 13 weeks = 650 K rows — DuckDB handles this in
#   milliseconds because it never writes to disk.
#
# CTE 4 — activity_flags
#   For each (user, week) row, LEFT JOIN to the events table.
#   The join condition checks whether ANY event falls inside the
#   7-day window for that week:
#     event_date >= signup_date + week_num * 7
#     event_date <  signup_date + (week_num + 1) * 7
#   We use MAX(CASE WHEN … THEN 1 ELSE 0 END) so we get a
#   single 0/1 flag per (user, week) — 1 if the user was active
#   that week, 0 otherwise.
#
# CTE 5 — cohort_weekly
#   GROUP BY (signup_month, week_number).
#   cohort_size  = distinct users in this signup_month
#   active_users = users who had the flag = 1 this week
#   retention_pct = active / cohort_size * 100
#
# Final SELECT: clean up and order.

COHORT_SQL = f"""
-- ── CTE 1: tag every user with a YYYY-MM cohort label ────
WITH user_cohorts AS (
    SELECT
        user_id,
        signup_date,
        -- STRFTIME formats a date as a string.
        -- '%Y-%m' gives us "2025-08", "2025-09", etc.
        STRFTIME(signup_date, '%Y-%m')  AS signup_month
    FROM users
    WHERE signup_date IS NOT NULL
),

-- ── CTE 2: generate week numbers 0 → {MAX_WEEK} ──────────────
weeks_spine AS (
    SELECT UNNEST(RANGE(0, {MAX_WEEK + 1})) AS week_num
    -- RANGE(0, 13) produces [0,1,2,...,12]
    -- UNNEST turns that list into individual rows
),

-- ── CTE 3: every (user, week) combination ────────────────
user_weeks AS (
    SELECT
        uc.user_id,
        uc.signup_date,
        uc.signup_month,
        ws.week_num
    FROM user_cohorts uc
    CROSS JOIN weeks_spine ws
    -- CROSS JOIN: each user is paired with every week number.
    -- No join condition needed — we want ALL combinations.
),

-- ── CTE 4: was each user active in each week? ────────────
activity_flags AS (
    SELECT
        uw.signup_month,
        uw.week_num,
        uw.user_id,
        -- Check if ANY qualifying event falls within this week's window.
        -- MAX() of a 0/1 expression collapses multiple matching events
        -- into a single "yes" (1) or "no" (0) per user-week.
        MAX(
            CASE
                WHEN e.event_date >= uw.signup_date + INTERVAL (uw.week_num * 7) DAY
                 AND e.event_date <  uw.signup_date + INTERVAL ((uw.week_num + 1) * 7) DAY
                 AND e.event_type IN {ACTIVE_TYPES}
                THEN 1
                ELSE 0
            END
        ) AS was_active
    FROM user_weeks uw
    LEFT JOIN events e
        ON e.user_id = uw.user_id
        -- The date-range filter is in the CASE above.
        -- We still join broadly first so DuckDB can use indexes,
        -- then the CASE narrows the window.
    GROUP BY uw.signup_month, uw.week_num, uw.user_id
),

-- ── CTE 5: aggregate to cohort level ─────────────────────
cohort_weekly AS (
    SELECT
        signup_month,
        week_num                              AS week_number,
        COUNT(DISTINCT user_id)               AS cohort_size,
        SUM(was_active)                       AS active_users,
        ROUND(
            SUM(was_active) * 100.0 / NULLIF(COUNT(DISTINCT user_id), 0),
            1
        )                                     AS retention_pct
    FROM activity_flags
    GROUP BY signup_month, week_num
)

-- ── Final result ──────────────────────────────────────────
SELECT
    signup_month,
    week_number,
    cohort_size,
    CAST(active_users AS INTEGER)  AS active_users,
    retention_pct
FROM cohort_weekly
ORDER BY signup_month, week_number
"""

print("   Running SQL cohort query …")
long_df = con.execute(COHORT_SQL).df()
con.close()

# ── Quick sanity check ────────────────────────────────────
n_cohorts = long_df["signup_month"].nunique()
print(f"   ✓ {len(long_df):,} rows  ·  {n_cohorts} monthly cohorts  ·  weeks 0–{MAX_WEEK}")

long_df.to_csv(LONG_CSV, index=False)
print(f"   ✓ Saved → {LONG_CSV}")

# ── Print a preview of the first cohort ──────────────────
first_cohort = long_df["signup_month"].min()
preview = long_df[long_df["signup_month"] == first_cohort]
print(f"\n   Preview — cohort {first_cohort}:")
print(preview[["week_number","cohort_size","active_users","retention_pct"]].to_string(index=False))

# ─────────────────────────────────────────────────────────
# STEP 2 — PIVOT TO WIDE MATRIX
# ─────────────────────────────────────────────────────────
print()
print("▶  STEP 2 — Pivoting to cohort matrix …")

# pivot_table:
#   rows   (index)   = signup_month
#   columns           = week_number (0, 1, 2 … 12)
#   values            = retention_pct
#   aggfunc           = first  (each cell already has one value)
matrix = long_df.pivot_table(
    index   = "signup_month",
    columns = "week_number",
    values  = "retention_pct",
    aggfunc = "first",
)

# Rename columns from integers to "Week 0", "Week 1" etc.
matrix.columns = [f"Week {int(c)}" for c in matrix.columns]
matrix.index.name = "signup_month"

# Sort rows chronologically (oldest cohort first = top row)
matrix = matrix.sort_index()

# ── Handle incompleteness ─────────────────────────────────
# Recent cohorts haven't had 12 weeks of history yet.
# Their future weeks will be NaN.  We leave NaN as-is for the
# heatmap — seaborn will render those cells as grey/blank,
# which is exactly what a real cohort chart does ("not yet due").

matrix.to_csv(MATRIX_CSV)
print(f"   ✓ Matrix shape: {matrix.shape[0]} cohorts × {matrix.shape[1]} weeks")
print(f"   ✓ Saved → {MATRIX_CSV}")

# ─────────────────────────────────────────────────────────
# STEP 3 — COHORT HEATMAP
# ─────────────────────────────────────────────────────────
print()
print("▶  STEP 3 — Drawing cohort heatmap …")

# ── Colour map: dark blue-green = high retention ──────────
# We build a custom LinearSegmentedColormap that goes from
# near-white (low retention) to deep teal (high retention).
# This is more readable than the default seaborn palettes.
teal_cmap = LinearSegmentedColormap.from_list(
    "teal_retention",
    ["#EAF4F4", "#A8D8D8", "#4AADAD", "#1A7A7A", "#0D4F4F"],
    N=256,
)

# ── Figure sizing ─────────────────────────────────────────
n_rows  = matrix.shape[0]         # one row per cohort month
n_cols  = matrix.shape[1]         # Week 0 … Week 12
fig_w   = max(16, n_cols * 1.1)
fig_h   = max(7,  n_rows * 0.62)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("#FAFBFC")
ax.set_facecolor("#FAFBFC")

# ── Build annotation array ────────────────────────────────
# For cells that have a value → "45%"
# For NaN cells (future weeks of recent cohorts) → "—"
# Build annotation strings element-wise — works on all pandas versions
annot_array = matrix.apply(
    lambda col: col.map(lambda v: f"{v:.0f}%" if pd.notna(v) else "—")
)

# ── Draw the heatmap ──────────────────────────────────────
sns.heatmap(
    matrix,
    ax          = ax,
    cmap        = teal_cmap,
    annot       = annot_array,     # use our custom "45%" labels
    fmt         = "",              # fmt must be "" when annot is a DataFrame of strings
    linewidths  = 0.6,
    linecolor   = "#FFFFFF",
    vmin        = 0,
    vmax        = 100,
    cbar_kws    = {
        "label"      : "Retention %",
        "shrink"     : 0.7,
        "pad"        : 0.02,
        "ticks"      : [0, 20, 40, 60, 80, 100],
    },
    annot_kws   = {"size": 8.5, "weight": "bold", "color": "#1C2B3A"},
    mask        = matrix.isna(),   # leave NaN cells visually blank
)

# ── Overlay grey "not yet" cells for NaN positions ────────
# seaborn's mask leaves them white; we fill them with a
# light-grey patch to clearly signal "data not yet available".
for r_idx in range(n_rows):
    for c_idx in range(n_cols):
        if pd.isna(matrix.iloc[r_idx, c_idx]):
            ax.add_patch(plt.Rectangle(
                (c_idx, r_idx), 1, 1,
                fill=True, color="#E8E8E8", zorder=2,
            ))
            ax.text(
                c_idx + 0.5, r_idx + 0.5, "—",
                ha="center", va="center",
                fontsize=8, color="#AAAAAA", zorder=3,
            )

# ── Axes formatting ───────────────────────────────────────
ax.set_title(
    "90-Day Retention Cohorts — Mutual Fund Users",
    fontsize    = 15,
    fontweight  = "bold",
    color       = "#1C2B3A",
    pad         = 18,
)
ax.set_xlabel("Weeks Since Signup", fontsize=11, labelpad=10, color="#3A4A5A")
ax.set_ylabel("Signup Month (Cohort)", fontsize=11, labelpad=10, color="#3A4A5A")

# Rotate x-axis labels to horizontal (default is angled)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

# ── Add a footnote ────────────────────────────────────────
fig.text(
    0.5, -0.01,
    "Grey cells (—) = weeks not yet reached for that cohort  |  "
    "Active = any investment event in the 7-day window",
    ha="center", fontsize=8, color="#7A8A9A",
)

plt.tight_layout()
plt.savefig(HEATMAP_PNG, dpi=150, bbox_inches="tight", facecolor="#FAFBFC")
plt.close()
print(f"   ✓ Saved → {HEATMAP_PNG}")

# ─────────────────────────────────────────────────────────
# STEP 4 — KEY METRICS
# ─────────────────────────────────────────────────────────
print()
print("▶  STEP 4 — Key retention metrics")
print()

# Helper: average retention for a given week across all cohorts
# that have valid (non-NaN) data for that week.
def avg_retention(wk_label):
    col = matrix[wk_label]
    valid = col.dropna()
    return valid.mean() if len(valid) else float("nan")

w1_avg  = avg_retention("Week 1")
w4_avg  = avg_retention("Week 4")
w12_avg = avg_retention("Week 12")

print(f"   Average Week-1  retention across all cohorts : {w1_avg:.1f}%")
print(f"   Average Week-4  retention across all cohorts : {w4_avg:.1f}%")
print(f"   Average Week-12 retention across all cohorts : {w12_avg:.1f}%")

# Best / worst cohort by Week-12 retention
# Drop cohorts where Week 12 is NaN (not enough history yet)
w12_col = matrix["Week 12"].dropna()

if len(w12_col) >= 1:
    best_cohort  = w12_col.idxmax()
    worst_cohort = w12_col.idxmin()
    print()
    print(f"   🏆 Best  Week-12 cohort : {best_cohort}  ({w12_col[best_cohort]:.1f}%)")
    print(f"   ⚠  Worst Week-12 cohort : {worst_cohort} ({w12_col[worst_cohort]:.1f}%)")
else:
    print()
    print("   (Not enough cohorts have reached Week 12 yet for best/worst comparison)")

# ── Retention waterfall: show the average across all cohorts ──
print()
print("   Average retention by week (all cohorts combined):")
print()
print(f"   {'Week':<10}  {'Avg Retention':>15}")
print(f"   {'────':<10}  {'─────────────':>15}")

for wk in range(MAX_WEEK + 1):
    label = f"Week {wk}"
    col   = matrix[label].dropna()
    if len(col) == 0:
        continue
    avg = col.mean()
    bar = "█" * int(avg / 3)    # ASCII bar scaled to fit terminal width
    print(f"   {label:<10}  {avg:>6.1f}%   {bar}")

# ─────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  ALL DONE — Output files")
print("=" * 60)
print(f"  Long CSV      : {LONG_CSV}")
print(f"  Matrix CSV    : {MATRIX_CSV}")
print(f"  Heatmap PNG   : {HEATMAP_PNG}")
print()