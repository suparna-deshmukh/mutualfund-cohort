"""
04_rfm.py
RFM (Recency, Frequency, Monetary) segmentation for mutual fund investors.

Steps:
  1. Compute raw R, F, M values per user via SQL
  2. Score each dimension 1–5 using quintile cuts
  3. Map scores to named segments
  4. Save rfm_segments.csv
  5. Print summary table
  6. Draw segment bar chart
  7. Print plain-English descriptions for At Risk / Hibernating / Lost

Author : Suparna Deshmukh
Project: mutualfund-cohort
"""

# ─────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────
import os
import duckdb
import pandas  as pd
import numpy   as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────
BASE_DIR      = "/Users/suparna_deshmukh/projects/mutualfund-cohort"
DB_PATH       = os.path.join(BASE_DIR, "data/mf.duckdb")

OUT_DIR       = os.path.join(BASE_DIR, "outputs")
FIG_DIR       = os.path.join(OUT_DIR,  "figures")
RFM_CSV       = os.path.join(OUT_DIR,  "rfm_segments.csv")
CHART_PNG     = os.path.join(FIG_DIR,  "rfm_segments.png")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
# Only count investment events — not redemptions
INVEST_TYPES  = "('first_investment','lumpsum','sip_setup','sip_installment')"

# Frequency and Monetary look back 180 days from the dataset's max date
LOOKBACK_DAYS = 180

print()
print("=" * 60)
print("  04_rfm.py — RFM Investor Segmentation")
print("=" * 60)

# ─────────────────────────────────────────────────────────
# STEP 1 — COMPUTE RAW R, F, M VALUES WITH SQL
# ─────────────────────────────────────────────────────────
print()
print("▶  STEP 1 — Computing raw RFM values …")

con = duckdb.connect(DB_PATH, read_only=True)

# ── How the SQL works ─────────────────────────────────────
#
# CTE: ref_date
#   The "snapshot date" — the most recent event_date in the
#   entire dataset.  All recency calculations are relative to
#   this date, not today's calendar date.  This makes the
#   analysis fully reproducible regardless of when you run it.
#
# CTE: invest_events
#   Filter the events table to investment-only rows (no redemptions).
#
# CTE: recency_cte
#   Per user: how many days since their last investment event?
#   DATE_DIFF('day', last_event, ref_date) gives a positive integer.
#   Smaller = more recent = better → scored in reverse below.
#
# CTE: freq_monetary_cte
#   Per user: count events AND sum amounts WITHIN the last 180 days.
#   We use a WHERE date filter rather than a window function so the
#   logic is easy to read and audit.
#
# Final SELECT: LEFT JOIN all three so every investor appears,
#   even if they had zero activity in the last 180 days (F=0, M=0).

RFM_SQL = f"""
-- ── snapshot date ────────────────────────────────────────
WITH ref_date AS (
    SELECT MAX(event_date) AS today
    FROM events
),

-- ── investment-only rows ─────────────────────────────────
invest_events AS (
    SELECT user_id, event_date, amount
    FROM events
    WHERE event_type IN {INVEST_TYPES}
),

-- ── Recency: days since last investment (all-time) ────────
recency_cte AS (
    SELECT
        ie.user_id,
        DATE_DIFF('day', MAX(ie.event_date), rd.today) AS recency_days
    FROM invest_events ie
    CROSS JOIN ref_date rd
    GROUP BY ie.user_id, rd.today
),

-- ── Frequency + Monetary: last {LOOKBACK_DAYS} days only ─────────
freq_monetary_cte AS (
    SELECT
        ie.user_id,
        COUNT(*)        AS frequency,
        SUM(ie.amount)  AS monetary
    FROM invest_events ie
    CROSS JOIN ref_date rd
    WHERE ie.event_date >= rd.today - INTERVAL {LOOKBACK_DAYS} DAY
    GROUP BY ie.user_id
)

-- ── Join all three: every investor gets R, F, M ──────────
SELECT
    r.user_id,
    r.recency_days,
    COALESCE(fm.frequency, 0)    AS frequency,
    COALESCE(fm.monetary,  0.0)  AS monetary
FROM recency_cte r
LEFT JOIN freq_monetary_cte fm ON r.user_id = fm.user_id
ORDER BY r.user_id
"""

rfm_df = con.execute(RFM_SQL).df()
con.close()

total_investors = len(rfm_df)
print(f"   ✓ {total_investors:,} investors with raw RFM values")
print(f"   Recency  — min: {rfm_df.recency_days.min():.0f}d  "
      f"median: {rfm_df.recency_days.median():.0f}d  "
      f"max: {rfm_df.recency_days.max():.0f}d")
print(f"   Frequency— median: {rfm_df.frequency.median():.1f}  "
      f"max: {rfm_df.frequency.max():.0f}")
print(f"   Monetary — median: ₹{rfm_df.monetary.median():,.0f}  "
      f"max: ₹{rfm_df.monetary.max():,.0f}")

# ─────────────────────────────────────────────────────────
# STEP 2 — SCORE R, F, M FROM 1 TO 5 USING QUINTILES
# ─────────────────────────────────────────────────────────
print()
print("▶  STEP 2 — Scoring R, F, M on 1–5 quintile scale …")

# ── What is a quintile? ───────────────────────────────────
# pd.qcut divides the data into 5 equal-sized buckets.
# The bottom 20% of values → label 1.
# The top    20% of values → label 5.
#
# R is REVERSED (lower recency days = more recent = better score):
#   pd.qcut assigns labels in ascending order by default.
#   We flip by using labels=[5,4,3,2,1] so the users who
#   invested most recently (fewest days ago) get score 5.
#
# F and M are normal (higher = better = higher score):
#   labels=[1,2,3,4,5]
#
# duplicates='drop' handles ties gracefully — if many users have
# the same value, qcut merges duplicate bin edges rather than
# crashing.

rfm_df["R"] = pd.qcut(
    rfm_df["recency_days"],
    q          = 5,
    labels     = [5, 4, 3, 2, 1],    # reversed: fewer days = score 5
    duplicates = "drop",
).astype(int)

rfm_df["F"] = pd.qcut(
    rfm_df["frequency"].rank(method="first"),   # rank() breaks ties
    q          = 5,
    labels     = [1, 2, 3, 4, 5],
    duplicates = "drop",
).astype(int)

rfm_df["M"] = pd.qcut(
    rfm_df["monetary"].rank(method="first"),
    q          = 5,
    labels     = [1, 2, 3, 4, 5],
    duplicates = "drop",
).astype(int)

print(f"   ✓ Scores assigned  —  R distribution:")
print(f"     {rfm_df['R'].value_counts().sort_index().to_dict()}")

# ─────────────────────────────────────────────────────────
# STEP 3 — MAP SCORES TO NAMED SEGMENTS
# ─────────────────────────────────────────────────────────
print()
print("▶  STEP 3 — Assigning segment labels …")

# ── Segment rules (applied in priority order) ────────────
# We evaluate conditions top-to-bottom and assign the FIRST
# matching segment.  Order matters:
#   Champions is checked before Loyal, so a user who qualifies
#   for both gets "Champions" (the more specific label).

def assign_segment(row):
    R, F = row["R"], row["F"]

    if   R >= 4 and F >= 4:               return "Champions"
    elif R >= 3 and F >= 3:               return "Loyal"
    elif R >= 3 and F <= 3:               return "Potential Loyalist"
    elif R >= 4 and F <= 2:               return "New Investor"
    elif 2 <= R <= 3 and F >= 3:          return "At Risk"
    elif 1 <= R <= 2 and 2 <= F <= 3:     return "Hibernating"
    else:                                  return "Lost"           # R 1–2, F 1–2

rfm_df["segment"] = rfm_df.apply(assign_segment, axis=1)

seg_counts = rfm_df["segment"].value_counts()
print(f"   ✓ Segments assigned:")
for seg, n in seg_counts.items():
    print(f"     {seg:<22}  {n:>6,} users  ({n/total_investors*100:.1f}%)")

# ─────────────────────────────────────────────────────────
# STEP 4 — SAVE rfm_segments.csv
# ─────────────────────────────────────────────────────────
print()
print("▶  STEP 4 — Saving RFM segments CSV …")

output_df = rfm_df[["user_id", "R", "F", "M",
                     "recency_days", "frequency", "monetary", "segment"]].copy()
output_df.rename(columns={
    "recency_days" : "recency_days_raw",
    "frequency"    : "frequency_raw",
    "monetary"     : "monetary_raw",
}, inplace=True)
output_df.to_csv(RFM_CSV, index=False)
print(f"   ✓ Saved → {RFM_CSV}")

# ─────────────────────────────────────────────────────────
# STEP 5 — PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────
print()
print("▶  STEP 5 — Segment summary")
print()

summary = (
    rfm_df
    .groupby("segment", observed=True)
    .agg(
        user_count      = ("user_id",      "count"),
        avg_recency_d   = ("recency_days", "mean"),
        avg_frequency   = ("frequency",    "mean"),
        avg_monetary_rs = ("monetary",     "mean"),
    )
    .reset_index()
)

# Custom sort so the summary reads from best to worst
SEG_ORDER = ["Champions", "Loyal", "Potential Loyalist",
             "New Investor", "At Risk", "Hibernating", "Lost"]
summary["_order"] = summary["segment"].map(
    {s: i for i, s in enumerate(SEG_ORDER)}
)
summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

summary["avg_recency_d"]   = summary["avg_recency_d"].round(0).astype(int)
summary["avg_frequency"]   = summary["avg_frequency"].round(1)
summary["avg_monetary_rs"] = summary["avg_monetary_rs"].round(0).astype(int)

# Pretty-print the table
col_widths = [22, 10, 14, 13, 18]
header = (f"{'Segment':<{col_widths[0]}}"
          f"{'Users':>{col_widths[1]}}"
          f"{'Avg Recency':>{col_widths[2]}}"
          f"{'Avg Freq':>{col_widths[3]}}"
          f"{'Avg Monetary':>{col_widths[4]}}")
divider = "─" * sum(col_widths)
print(f"   {header}")
print(f"   {divider}")

for _, row in summary.iterrows():
    line = (f"   {row['segment']:<{col_widths[0]}}"
            f"{row['user_count']:>{col_widths[1]},}"
            f"{str(row['avg_recency_d'])+'d':>{col_widths[2]}}"
            f"{row['avg_frequency']:>{col_widths[3]}.1f}"
            f"{'₹'+format(row['avg_monetary_rs'], ','):>{col_widths[4]}}")
    print(line)

print(f"   {divider}")

# ─────────────────────────────────────────────────────────
# STEP 6 — SEGMENT BAR CHART
# ─────────────────────────────────────────────────────────
print()
print("▶  STEP 6 — Drawing segment bar chart …")

# ── One colour per segment ────────────────────────────────
# Colours are semantically meaningful:
#   green  → healthy segments   (Champions, Loyal, Potential, New)
#   amber  → warning segments   (At Risk)
#   grey   → dormant segments   (Hibernating, Lost)
SEGMENT_COLOURS = {
    "Champions"         : "#1A7A4A",   # deep green
    "Loyal"             : "#2EAD6E",   # medium green
    "Potential Loyalist": "#74C99A",   # light green
    "New Investor"      : "#5BAED6",   # sky blue
    "At Risk"           : "#F0A500",   # amber
    "Hibernating"       : "#C07830",   # warm brown
    "Lost"              : "#9A9A9A",   # grey
}

# Build the data in the canonical order
plot_segs   = [s for s in SEG_ORDER if s in seg_counts.index]
plot_counts = [seg_counts[s] for s in plot_segs]
plot_colors = [SEGMENT_COLOURS[s] for s in plot_segs]

BACKGROUND = "#F7F9FC"
TEXT_DARK   = "#1C2B3A"

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BACKGROUND)
ax.set_facecolor(BACKGROUND)

bars = ax.bar(
    plot_segs,
    plot_counts,
    color       = plot_colors,
    width       = 0.62,
    edgecolor   = "white",
    linewidth   = 1.2,
    zorder      = 3,
)

# ── Annotate each bar with count + % ─────────────────────
for bar, count in zip(bars, plot_counts):
    pct = count / total_investors * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total_investors * 0.004,
        f"{count:,}\n({pct:.1f}%)",
        ha         = "center",
        va         = "bottom",
        fontsize   = 9,
        fontweight = "bold",
        color      = TEXT_DARK,
        linespacing= 1.4,
    )

# ── Grid lines behind bars ────────────────────────────────
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8, linestyle="--")
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="both", which="both", length=0)

# ── Labels & title ────────────────────────────────────────
ax.set_xlabel("Investor Segment", fontsize=11, labelpad=10, color=TEXT_DARK)
ax.set_ylabel("Number of Users",  fontsize=11, labelpad=10, color=TEXT_DARK)
ax.set_title(
    "RFM Investor Segmentation — Mutual Fund Users",
    fontsize   = 14,
    fontweight = "bold",
    color      = TEXT_DARK,
    pad        = 18,
)
ax.set_xticks(range(len(plot_segs)))
ax.set_xticklabels(plot_segs, rotation=15, ha="right", fontsize=10)
ax.set_ylim(0, max(plot_counts) * 1.18)

# ── Subtle colour legend in top-right ─────────────────────
legend_handles = [
    mpatches.Patch(color=SEGMENT_COLOURS[s], label=s)
    for s in plot_segs
]
ax.legend(
    handles       = legend_handles,
    loc           = "upper right",
    frameon       = True,
    framealpha    = 0.85,
    fontsize      = 8.5,
    edgecolor     = "#CCCCCC",
    title         = "Segment",
    title_fontsize= 9,
)

# ── Footnote ─────────────────────────────────────────────
fig.text(
    0.5, -0.02,
    "RFM scored 1–5 by quintile  ·  F & M based on last 180 days  "
    "·  R = days since last investment",
    ha="center", fontsize=8, color="#7A8A9A",
)

plt.tight_layout()
plt.savefig(CHART_PNG, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
plt.close()
print(f"   ✓ Saved → {CHART_PNG}")

# ─────────────────────────────────────────────────────────
# STEP 7 — PLAIN-ENGLISH DESCRIPTIONS FOR AT-RISK SEGMENTS
# ─────────────────────────────────────────────────────────
print()
print("▶  STEP 7 — Segment health check")
print()

def seg_stats(name):
    """Return a dict of avg stats for the named segment."""
    sub = rfm_df[rfm_df["segment"] == name]
    return {
        "n"    : len(sub),
        "r"    : sub["recency_days"].mean(),
        "f"    : sub["frequency"].mean(),
        "m"    : sub["monetary"].mean(),
    }

for seg_name in ["At Risk", "Hibernating", "Lost"]:
    s = seg_stats(seg_name)
    if s["n"] == 0:
        continue

    if seg_name == "At Risk":
        desc = (
            f"⚠  AT RISK  ({s['n']:,} users)\n"
            f"   These investors used to be frequent buyers (avg {s['f']:.1f} events/180d)\n"
            f"   but haven't invested in ~{s['r']:.0f} days on average. They still have\n"
            f"   decent spending power (avg ₹{s['m']:,.0f} in last 180d) but engagement\n"
            f"   is cooling. A targeted re-activation campaign (e.g. SIP top-up nudge,\n"
            f"   market-update email) within the next 2–3 weeks is recommended."
        )
    elif seg_name == "Hibernating":
        desc = (
            f"😴  HIBERNATING  ({s['n']:,} users)\n"
            f"   Low recent activity (avg {s['f']:.1f} events in last 180d) and last\n"
            f"   invested ~{s['r']:.0f} days ago. Their monetary contribution has\n"
            f"   dropped to ₹{s['m']:,.0f} on average. These users are not yet lost —\n"
            f"   a 'We miss you' campaign with a low-commitment offer (e.g. ₹500 SIP)\n"
            f"   can still re-engage a meaningful fraction."
        )
    elif seg_name == "Lost":
        desc = (
            f"❌  LOST  ({s['n']:,} users)\n"
            f"   Haven't invested in ~{s['r']:.0f} days on average and showed minimal\n"
            f"   activity even when active (avg {s['f']:.1f} events/180d, ₹{s['m']:,.0f}).\n"
            f"   Win-back ROI is low for this group. Focus energy on preventing\n"
            f"   'At Risk' users from sliding here. If budget allows, a single\n"
            f"   low-cost re-engagement SMS (e.g. a market milestone) is the max\n"
            f"   recommended investment for this segment."
        )

    print(desc)
    print()

# ─────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("  ALL DONE — Output files")
print("=" * 60)
print(f"  RFM CSV   : {RFM_CSV}")
print(f"  Chart PNG : {CHART_PNG}")
print()