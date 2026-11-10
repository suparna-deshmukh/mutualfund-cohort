"""
02_funnel.py
Activation-funnel analysis for the mutual-fund cohort project.

Steps:
  1. Load users.csv + events.csv into DuckDB
  2. Run a 6-stage activation-funnel SQL query
  3. Draw a horizontal funnel bar chart
  4. Break funnel Stage 3 down by source_channel

Author : Suparna Deshmukh
Project: mutualfund-cohort
"""

# ─────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────
import os
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

# ─────────────────────────────────────────────────────────
# PATHS  — all inputs and outputs live here
# ─────────────────────────────────────────────────────────
BASE_DIR    = "/Users/suparna_deshmukh/projects/mutualfund-cohort"

USERS_CSV   = os.path.join(BASE_DIR, "data/users.csv")
EVENTS_CSV  = os.path.join(BASE_DIR, "data/events.csv")
DB_PATH     = os.path.join(BASE_DIR, "data/mf.duckdb")

OUT_DIR     = os.path.join(BASE_DIR, "outputs")
FIG_DIR     = os.path.join(OUT_DIR,  "figures")
FUNNEL_CSV  = os.path.join(OUT_DIR,  "activation_funnel.csv")
CHANNEL_CSV = os.path.join(OUT_DIR,  "funnel_by_channel.csv")
FUNNEL_PNG  = os.path.join(FIG_DIR,  "funnel.png")

# Create output folders if they don't exist yet
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA INTO DUCKDB
# ─────────────────────────────────────────────────────────
print()
print("=" * 56)
print("  STEP 1 — Loading CSVs into DuckDB")
print("=" * 56)

# duckdb.connect() creates the .duckdb file if it doesn't exist.
# We pass read_only=False (the default) so we can write tables.
con = duckdb.connect(DB_PATH)

# Drop tables so the script is safe to re-run
con.execute("DROP TABLE IF EXISTS users")
con.execute("DROP TABLE IF EXISTS events")

# read_csv_auto lets DuckDB sniff column types automatically
con.execute(f"""
    CREATE TABLE users AS
    SELECT * FROM read_csv_auto('{USERS_CSV}', header=true)
""")

con.execute(f"""
    CREATE TABLE events AS
    SELECT * FROM read_csv_auto('{EVENTS_CSV}', header=true)
""")

users_n  = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
events_n = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
print(f"  ✓ users  table : {users_n:>10,} rows")
print(f"  ✓ events table : {events_n:>10,} rows")

# ─────────────────────────────────────────────────────────
# STEP 2 — 6-STAGE ACTIVATION FUNNEL (SQL with CTEs)
# ─────────────────────────────────────────────────────────
print()
print("=" * 56)
print("  STEP 2 — Computing activation funnel")
print("=" * 56)

# ── How CTEs (WITH statements) work ────────────────────
# Each WITH block is a named temporary result set.
# Later blocks can reference earlier ones by name.
# Think of them like numbered steps in a recipe.
#
# Stage definitions:
#   S1  — every user who signed up
#   S2  — users with a non-null kyc_completed_date
#   S3  — users who appear in events with event_type = 'first_investment'
#   S4  — users with a 'sip_setup' event
#   S5  — users who made ANY investment within 30 days AFTER their first
#   S6  — users with at least one event in the window [day 75, day 105]
#          (centred on day 90 since signup)

FUNNEL_SQL = """
-- ── CTE 1: total signups ──────────────────────────────
WITH s1 AS (
    SELECT COUNT(DISTINCT user_id) AS n
    FROM users
),

-- ── CTE 2: KYC completed ──────────────────────────────
s2 AS (
    SELECT COUNT(DISTINCT user_id) AS n
    FROM users
    WHERE kyc_completed_date IS NOT NULL
),

-- ── CTE 3: made a first investment ───────────────────
-- We look for rows in events where event_type = 'first_investment'
s3 AS (
    SELECT COUNT(DISTINCT user_id) AS n
    FROM events
    WHERE event_type = 'first_investment'
),

-- ── CTE 4: set up an SIP ─────────────────────────────
s4 AS (
    SELECT COUNT(DISTINCT user_id) AS n
    FROM events
    WHERE event_type = 'sip_setup'
),

-- ── CTE 5: second investment within 30 days ──────────
-- First, find each user's first-investment date.
-- Then check if that same user has ANY other investment
-- (lumpsum, sip_installment, or another first_investment)
-- within the next 30 days.
first_inv AS (
    SELECT user_id,
           MIN(event_date) AS first_date
    FROM events
    WHERE event_type = 'first_investment'
    GROUP BY user_id
),
s5 AS (
    SELECT COUNT(DISTINCT e.user_id) AS n
    FROM events e
    JOIN first_inv fi ON e.user_id = fi.user_id
    WHERE e.event_type IN ('lumpsum', 'sip_installment', 'sip_setup')
      AND e.event_date > fi.first_date
      AND e.event_date <= fi.first_date + INTERVAL 30 DAY
),

-- ── CTE 6: still active at day 90 ────────────────────
-- "Active at day 90" = had at least one event between
-- day 75 and day 105 since signup (a 30-day window
-- centred on day 90 — gives a fair check without being
-- too strict about hitting exactly day 90).
s6 AS (
    SELECT COUNT(DISTINCT e.user_id) AS n
    FROM events e
    JOIN users u ON e.user_id = u.user_id
    WHERE CAST(e.event_date AS DATE)
              BETWEEN CAST(u.signup_date AS DATE) + INTERVAL 75 DAY
                  AND CAST(u.signup_date AS DATE) + INTERVAL 105 DAY
)

-- ── Final SELECT: assemble into one result table ─────
-- We use UNION ALL to stack each stage as its own row.
SELECT 1  AS stage_num, 'Signed Up'                   AS stage_name, (SELECT n FROM s1) AS user_count
UNION ALL
SELECT 2, 'Completed KYC',                                            (SELECT n FROM s2)
UNION ALL
SELECT 3, 'Made First Investment',                                     (SELECT n FROM s3)
UNION ALL
SELECT 4, 'Set Up SIP',                                                (SELECT n FROM s4)
UNION ALL
SELECT 5, 'Second Investment ≤30d',                                    (SELECT n FROM s5)
UNION ALL
SELECT 6, 'Active at Day 90',                                          (SELECT n FROM s6)
ORDER BY stage_num
"""

funnel_df = con.execute(FUNNEL_SQL).df()   # .df() returns a pandas DataFrame

# ── Compute derived columns in Python ────────────────
# conversion_from_prev : how many of the previous stage reached this one
# conversion_from_signup : how many of ALL signups reached this stage
top = funnel_df.loc[0, "user_count"]        # Stage 1 count (total signups)

conv_prev   = []
conv_signup = []

for i, row in funnel_df.iterrows():
    # Conversion from previous stage
    if i == 0:
        conv_prev.append(100.0)             # Stage 1 is the baseline
    else:
        prev = funnel_df.loc[i - 1, "user_count"]
        pct  = row["user_count"] / prev * 100 if prev else 0
        conv_prev.append(round(pct, 1))

    # Overall conversion from Stage 1
    pct_top = row["user_count"] / top * 100 if top else 0
    conv_signup.append(round(pct_top, 1))

funnel_df["conv_from_prev_pct"]   = conv_prev
funnel_df["conv_from_signup_pct"] = conv_signup

# Save
funnel_df.to_csv(FUNNEL_CSV, index=False)
print()
print(funnel_df.to_string(index=False))
print()
print(f"  ✓ Saved → {FUNNEL_CSV}")

# ─────────────────────────────────────────────────────────
# STEP 3 — HORIZONTAL FUNNEL BAR CHART
# ─────────────────────────────────────────────────────────
print()
print("=" * 56)
print("  STEP 3 — Drawing funnel chart")
print("=" * 56)

# ── Colour palette ────────────────────────────────────
# A gradient from deep blue → teal → green to represent
# users moving through the funnel.
BAR_COLORS = [
    "#1A4E8C",   # Stage 1 — signed up     (deep blue)
    "#2471B5",   # Stage 2 — KYC           (blue)
    "#2E9AC4",   # Stage 3 — first invest  (sky)
    "#27A99A",   # Stage 4 — SIP           (teal)
    "#2DBF7E",   # Stage 5 — second invest (mint)
    "#34C95A",   # Stage 6 — active 90d    (green)
]
BACKGROUND = "#F7F9FC"
TEXT_DARK   = "#1C2B3A"
TEXT_LIGHT  = "#FFFFFF"
DROP_COLOR  = "#E85454"    # red for drop-off annotations

# ── Layout ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor(BACKGROUND)
ax.set_facecolor(BACKGROUND)

stages    = funnel_df["stage_name"].tolist()
counts    = funnel_df["user_count"].tolist()
max_count = counts[0]

# Horizontal bars: widths are proportional to user count.
# We centre each bar so the chart looks like a proper funnel.
bar_height = 0.55      # fraction of the row height each bar occupies

for i, (stage, count, color) in enumerate(zip(stages, counts, BAR_COLORS)):
    y       = len(stages) - 1 - i       # flip so Stage 1 is at the top
    width   = count / max_count         # fraction 0–1

    # ── Draw the bar (centred) ────────────────────────
    bar = FancyBboxPatch(
        xy=((1 - width) / 2, y - bar_height / 2),
        width=width,
        height=bar_height,
        boxstyle="round,pad=0.005",
        facecolor=color,
        edgecolor="none",
        zorder=3,
    )
    ax.add_patch(bar)

    # ── Stage label (left side) ───────────────────────
    ax.text(
        (1 - width) / 2 - 0.012, y,
        stage,
        ha="right", va="center",
        fontsize=10, color=TEXT_DARK,
        fontweight="bold",
    )

    # ── Count inside the bar ──────────────────────────
    ax.text(
        0.5, y,
        f"{count:,}",
        ha="center", va="center",
        fontsize=11, color=TEXT_LIGHT,
        fontweight="bold", zorder=4,
    )

    # ── Drop-off annotation between bars ─────────────
    # Between stage i and stage i+1, show how many users were lost
    # and the percentage drop from the previous stage.
    if i < len(stages) - 1:
        next_count  = counts[i + 1]
        drop_n      = count - next_count
        drop_pct    = drop_n / count * 100
        # Position the annotation in the gap between bars
        ax.text(
            0.50, y - 0.5,
            f"▼  {drop_pct:.1f}% drop  ({drop_n:,} users)",
            ha="center", va="center",
            fontsize=8.5, color=DROP_COLOR,
            style="italic", zorder=4,
        )

# ── Axes clean-up ─────────────────────────────────────
ax.set_xlim(0, 1)
ax.set_ylim(-0.8, len(stages) - 0.2)
ax.axis("off")                         # hide ticks/spines — we drew everything manually

# ── Title & subtitle ──────────────────────────────────
fig.text(
    0.5, 0.97,
    "Mutual Fund App — Activation Funnel",
    ha="center", va="top",
    fontsize=16, fontweight="bold", color=TEXT_DARK,
)
fig.text(
    0.5, 0.93,
    f"50,000 users  •  Overall signup → active-90d conversion: "
    f"{funnel_df.loc[5, 'conv_from_signup_pct']:.1f}%",
    ha="center", va="top",
    fontsize=10, color="#5A6A7A",
)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(FUNNEL_PNG, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
plt.close()
print(f"  ✓ Saved → {FUNNEL_PNG}")

# ─────────────────────────────────────────────────────────
# STEP 4 — CHANNEL BREAKDOWN (Stage 3 conversion)
# ─────────────────────────────────────────────────────────
print()
print("=" * 56)
print("  STEP 4 — Channel breakdown")
print("=" * 56)

# For each source_channel:
#   • Count how many users signed up via that channel
#   • Count how many of those users made a first investment
#   • Compute the conversion rate
CHANNEL_SQL = """
WITH channel_signups AS (
    SELECT source_channel,
           COUNT(DISTINCT user_id) AS signed_up
    FROM users
    GROUP BY source_channel
),
channel_investors AS (
    SELECT u.source_channel,
           COUNT(DISTINCT e.user_id) AS invested
    FROM events e
    JOIN users u ON e.user_id = u.user_id
    WHERE e.event_type = 'first_investment'
    GROUP BY u.source_channel
)
SELECT
    cs.source_channel,
    cs.signed_up,
    COALESCE(ci.invested, 0)                                     AS made_first_investment,
    ROUND(COALESCE(ci.invested, 0) * 100.0 / cs.signed_up, 1)   AS conversion_pct
FROM channel_signups cs
LEFT JOIN channel_investors ci ON cs.source_channel = ci.source_channel
ORDER BY conversion_pct DESC
"""

channel_df = con.execute(CHANNEL_SQL).df()

# ── Identify worst-performing channel ────────────────
worst_channel = channel_df.loc[channel_df["conversion_pct"].idxmin(), "source_channel"]
worst_pct     = channel_df["conversion_pct"].min()

channel_df.to_csv(CHANNEL_CSV, index=False)

print()
print(channel_df.to_string(index=False))
print()
print(f"  ⚠  Worst converting channel: '{worst_channel}' ({worst_pct}%)")
print()
print(f"  ✓ Saved → {CHANNEL_CSV}")

# ─────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────
print()
print("=" * 56)
print("  ALL DONE — Output files")
print("=" * 56)
print(f"  Database    : {DB_PATH}")
print(f"  Funnel CSV  : {FUNNEL_CSV}")
print(f"  Channel CSV : {CHANNEL_CSV}")
print(f"  Funnel PNG  : {FUNNEL_PNG}")
print()

con.close()