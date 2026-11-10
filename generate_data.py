"""
generate_data.py
Synthetic data generator for mutual fund cohort analytics.
Generates 50,000 users and their investment events.

Author: Suparna Deshmukh
Project: mutualfund-cohort
"""

# ─────────────────────────────────────────────
# IMPORTS — built-in Python libraries only,
# no installation needed except numpy/pandas
# ─────────────────────────────────────────────
import random
import math
import os
import csv
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ── Seed for reproducibility ──────────────────
# Using both random and numpy seeds so every run
# produces the EXACT same data.
random.seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────
# CONFIGURATION — change these if you ever want
# to tweak the simulation parameters
# ─────────────────────────────────────────────
NUM_USERS          = 50_000
TODAY              = date.today()
HISTORY_DAYS       = 548          # ~18 months back
START_DATE         = TODAY - timedelta(days=HISTORY_DAYS)

OUTPUT_DIR         = "/Users/suparna_deshmukh/projects/mutualfund-cohort/data"
USERS_CSV          = os.path.join(OUTPUT_DIR, "users.csv")
EVENTS_CSV         = os.path.join(OUTPUT_DIR, "events.csv")

# City-tier probabilities
CITY_TIER_CHOICES  = ["Tier 1", "Tier 2", "Tier 3"]
CITY_TIER_PROBS    = [0.40,      0.35,      0.25]

# Source-channel probabilities
CHANNEL_CHOICES    = ["organic", "referral", "paid_social", "paid_search", "partnership"]
CHANNEL_PROBS      = [0.30,      0.25,       0.20,          0.15,          0.10]

# Scheme categories for investment events
SCHEME_CATEGORIES  = ["Equity Large Cap", "Equity Mid Cap", "Equity Small Cap",
                       "Debt Short Term",  "Debt Long Term",  "Hybrid Balanced",
                       "ELSS",             "Index Fund",      "Liquid Fund"]

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def make_output_dir():
    """Create the output directory if it doesn't already exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def weighted_choice(choices, probs):
    """
    Pick one item from `choices` using the given probabilities.
    np.random.choice handles this cleanly.
    """
    return np.random.choice(choices, p=probs)


def generate_signup_dates(n, start_date, end_date):
    """
    Generate n signup dates between start_date and end_date with an
    EXPONENTIAL GROWTH pattern — more signups happen closer to today.

    How it works:
      1. We map each day in the range to a weight = e^(k * fraction_through_range)
         where fraction 0 = oldest date and fraction 1 = today.
      2. We normalise the weights so they sum to 1 (making them probabilities).
      3. We sample n days from this distribution (with replacement, since
         multiple users can sign up on the same day).
    """
    total_days = (end_date - start_date).days
    day_offsets = np.arange(total_days)                        # 0, 1, 2 … total_days-1
    fractions   = day_offsets / total_days                     # 0.0 … ~1.0
    k           = 3.5                                          # growth steepness factor
    weights     = np.exp(k * fractions)
    weights     = weights / weights.sum()                      # normalise to probabilities

    chosen_offsets = np.random.choice(day_offsets, size=n, p=weights)
    return [start_date + timedelta(days=int(d)) for d in chosen_offsets]


def generate_age():
    """
    Draw an age from a normal distribution centred at 32,
    clipped to the range [22, 58].
    """
    age = int(np.random.normal(loc=32, scale=7))
    return max(22, min(58, age))


def age_to_risk(age):
    """
    Map age to risk appetite.
    Younger users lean aggressive; older users lean conservative.

    Age bracket → probabilities for [conservative, moderate, aggressive]
      22–30  → 10%, 35%, 55%
      31–40  → 25%, 45%, 30%
      41–50  → 45%, 40%, 15%
      51–58  → 65%, 30%,  5%
    """
    if age <= 30:
        probs = [0.10, 0.35, 0.55]
    elif age <= 40:
        probs = [0.25, 0.45, 0.30]
    elif age <= 50:
        probs = [0.45, 0.40, 0.15]
    else:
        probs = [0.65, 0.30, 0.05]
    return weighted_choice(["conservative", "moderate", "aggressive"], probs)


def generate_kyc_date(signup_date, completed_kyc):
    """
    Return a KYC completion date (or None if the user never completed KYC).

    85% of users complete KYC.
    Of those:
      70% within  7 days of signup
      20% within 30 days of signup  (i.e. 8–30 days)
      10% after   30 days            (i.e. 31–180 days)
    """
    if not completed_kyc:
        return None

    r = random.random()
    if r < 0.70:
        # 1–7 days after signup
        delay = random.randint(1, 7)
    elif r < 0.90:
        # 8–30 days after signup
        delay = random.randint(8, 30)
    else:
        # 31–180 days after signup
        delay = random.randint(31, 180)

    kyc_date = signup_date + timedelta(days=delay)
    # KYC date must not be in the future
    return min(kyc_date, TODAY)


def lognormal_amount(mean_target, low, high):
    """
    Generate a rupee amount using a log-normal distribution.

    Log-normal is ideal for monetary values because:
      • It only produces positive numbers.
      • It has a long right tail (a few large transactions are realistic).

    We back-calculate the mu parameter so the distribution centres near
    `mean_target`, then clip to [low, high].
    """
    mu    = math.log(mean_target)
    sigma = 0.8           # controls how spread out values are
    amount = np.random.lognormal(mean=mu, sigma=sigma)
    return max(low, min(high, round(amount, -2)))   # round to nearest ₹100


def pick_scheme():
    """Pick a random scheme category for an investment event."""
    return random.choice(SCHEME_CATEGORIES)


# ─────────────────────────────────────────────
# STEP 1 — GENERATE USERS TABLE
# ─────────────────────────────────────────────

def build_users():
    """
    Build and return a list of user dictionaries.
    Each dict represents one row in users.csv.
    """
    print("▶  Generating users …")

    signup_dates = generate_signup_dates(NUM_USERS, START_DATE, TODAY)

    users = []
    for i in range(NUM_USERS):
        user_id     = f"U{i + 1:05d}"           # U00001 … U50000
        signup_date = signup_dates[i]
        age         = generate_age()
        city_tier   = weighted_choice(CITY_TIER_CHOICES, CITY_TIER_PROBS)
        channel     = weighted_choice(CHANNEL_CHOICES,   CHANNEL_PROBS)
        risk        = age_to_risk(age)

        # 85% chance of completing KYC
        completed_kyc = random.random() < 0.85
        kyc_date      = generate_kyc_date(signup_date, completed_kyc)

        users.append({
            "user_id":            user_id,
            "signup_date":        signup_date,
            "age":                age,
            "city_tier":          city_tier,
            "kyc_completed_date": kyc_date if kyc_date else "",
            "source_channel":     channel,
            "risk_appetite":      risk,
        })

    print(f"   ✓ {len(users):,} users generated")
    return users


# ─────────────────────────────────────────────
# STEP 2 — GENERATE EVENTS TABLE
# ─────────────────────────────────────────────

def build_events(users):
    """
    Build and return a list of event dictionaries.
    Each dict represents one row in events.csv.

    Investment logic per user:
      1. Only KYC-completed users can invest.
      2. 60% of KYC users make a first_investment within 14 days of KYC.
      3. 50% of first-investors set up a SIP.
      4. SIPs fire one sip_installment event each month.
      5. Dormancy: after first investment, users randomly stop investing.
         By 90 days  → 35% of investors are dormant (no new investment in 30+ days).
         By 180 days → 55% are dormant.
         We model this with a per-user "stop date" drawn from a mixture distribution.
      6. Non-dormant users also make occasional lumpsum investments.
    """
    print("▶  Generating events …")

    events        = []
    event_counter = 0          # used to create unique event_id values
    investor_count = 0
    dormant_90_count = 0

    for user in users:
        # ── Skip users who haven't completed KYC ──────────────────
        if not user["kyc_completed_date"]:
            continue

        kyc_date = user["kyc_completed_date"]
        if isinstance(kyc_date, str):
            kyc_date = date.fromisoformat(kyc_date)

        # ── 60% chance of first investment ────────────────────────
        if random.random() > 0.60:
            continue

        investor_count += 1

        # First investment happens 1–14 days after KYC
        first_invest_delay = random.randint(1, 14)
        first_invest_date  = kyc_date + timedelta(days=first_invest_delay)

        # Must not be in the future
        if first_invest_date > TODAY:
            continue

        # ── Determine this user's "go-dormant" date ───────────────
        # We model dormancy by assigning each investor a date after
        # which they make no more investments.
        #
        # We want:
        #   35% dormant by day 90  after first investment
        #   55% dormant by day 180 after first investment
        #
        # Strategy: draw a dormancy onset from a mixture of:
        #   • 35% of users  → stop between day 30 and day 90
        #   • 20% of users  → stop between day 90 and day 180  (35+20=55%)
        #   • 45% of users  → never go dormant within our window
        #     (represented as day 999 = effectively never)
        #
        r_dormancy = random.random()
        if r_dormancy < 0.35:
            # Goes dormant in the 30–90 day window
            dormant_offset = random.randint(30, 90)
        elif r_dormancy < 0.55:
            # Goes dormant in the 90–180 day window
            dormant_offset = random.randint(90, 180)
        else:
            # Stays active; use a large number so the cutoff is never hit
            dormant_offset = 999

        dormant_date = first_invest_date + timedelta(days=dormant_offset)

        # ── Record first_investment event ─────────────────────────
        event_counter += 1
        events.append({
            "event_id":        f"E{event_counter:08d}",
            "user_id":         user["user_id"],
            "event_date":      first_invest_date,
            "event_type":      "first_investment",
            "scheme_category": pick_scheme(),
            "amount":          lognormal_amount(25_000, 5_000, 2_00_000),
        })

        # ── SIP setup (50% of first-investors) ───────────────────
        has_sip   = random.random() < 0.50
        sip_amount = lognormal_amount(2_500, 500, 10_000) if has_sip else 0
        sip_scheme = pick_scheme() if has_sip else None

        if has_sip:
            sip_setup_date = first_invest_date + timedelta(days=random.randint(0, 3))
            if sip_setup_date <= TODAY and sip_setup_date < dormant_date:
                event_counter += 1
                events.append({
                    "event_id":        f"E{event_counter:08d}",
                    "user_id":         user["user_id"],
                    "event_date":      sip_setup_date,
                    "event_type":      "sip_setup",
                    "scheme_category": sip_scheme,
                    "amount":          sip_amount,
                })

                # Generate monthly SIP installments until dormancy or today
                installment_date = sip_setup_date + timedelta(days=30)
                while installment_date <= TODAY and installment_date < dormant_date:
                    event_counter += 1
                    events.append({
                        "event_id":        f"E{event_counter:08d}",
                        "user_id":         user["user_id"],
                        "event_date":      installment_date,
                        "event_type":      "sip_installment",
                        "scheme_category": sip_scheme,
                        "amount":          sip_amount,
                    })
                    installment_date += timedelta(days=30)

        # ── Occasional lumpsum investments (active users) ─────────
        # Active users make 0–3 additional lumpsum investments per year,
        # scattered randomly within their active window.
        active_window_days = (min(dormant_date, TODAY) - first_invest_date).days
        if active_window_days > 30:
            # Frequency: ~1 lumpsum every 4 months on average
            num_lumpsums = np.random.poisson(active_window_days / 120)
            for _ in range(num_lumpsums):
                offset = random.randint(15, active_window_days)
                ls_date = first_invest_date + timedelta(days=offset)
                if ls_date <= TODAY:
                    event_counter += 1
                    events.append({
                        "event_id":        f"E{event_counter:08d}",
                        "user_id":         user["user_id"],
                        "event_date":      ls_date,
                        "event_type":      "lumpsum",
                        "scheme_category": pick_scheme(),
                        "amount":          lognormal_amount(25_000, 5_000, 2_00_000),
                    })

        # ── Occasional redemptions (10% of active investors) ──────
        if random.random() < 0.10 and active_window_days > 60:
            redeem_offset = random.randint(60, active_window_days)
            redeem_date   = first_invest_date + timedelta(days=redeem_offset)
            if redeem_date <= TODAY:
                event_counter += 1
                events.append({
                    "event_id":        f"E{event_counter:08d}",
                    "user_id":         user["user_id"],
                    "event_date":      redeem_date,
                    "event_type":      "redemption",
                    "scheme_category": pick_scheme(),
                    "amount":          lognormal_amount(15_000, 1_000, 1_50_000),
                })

        # ── Track dormancy at 90 days for the summary stat ────────
        if dormant_offset <= 90:
            dormant_90_count += 1

    print(f"   ✓ {len(events):,} events generated for {investor_count:,} investors")
    return events, investor_count, dormant_90_count


# ─────────────────────────────────────────────
# STEP 3 — WRITE CSV FILES
# ─────────────────────────────────────────────

def write_users_csv(users):
    """Write the users list to users.csv."""
    fieldnames = ["user_id", "signup_date", "age", "city_tier",
                  "kyc_completed_date", "source_channel", "risk_appetite"]
    with open(USERS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(users)
    print(f"   ✓ Saved → {USERS_CSV}")


def write_events_csv(events):
    """Write the events list to events.csv."""
    # Sort events by date before saving — makes downstream analysis easier
    events_sorted = sorted(events, key=lambda e: (e["user_id"], e["event_date"]))

    fieldnames = ["event_id", "user_id", "event_date", "event_type",
                  "scheme_category", "amount"]
    with open(EVENTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events_sorted)
    print(f"   ✓ Saved → {EVENTS_CSV}")


# ─────────────────────────────────────────────
# STEP 4 — PRINT SUMMARY STATISTICS
# ─────────────────────────────────────────────

def print_summary(users, events, investor_count, dormant_90_count):
    """Print a clean summary to the terminal after generation."""

    total_users  = len(users)
    total_events = len(events)

    kyc_users    = sum(1 for u in users if u["kyc_completed_date"])
    pct_kyc      = kyc_users / total_users * 100

    pct_invested = investor_count / total_users * 100

    pct_dormant_90 = dormant_90_count / investor_count * 100 if investor_count else 0

    print()
    print("=" * 50)
    print("  DATA GENERATION COMPLETE")
    print("=" * 50)
    print(f"  Total users              : {total_users:>10,}")
    print(f"  Total events             : {total_events:>10,}")
    print(f"  KYC completion rate      : {pct_kyc:>9.1f}%")
    print(f"  Users who invested       : {pct_invested:>9.1f}%")
    print(f"  Investors dormant @90d   : {pct_dormant_90:>9.1f}%")
    print("=" * 50)
    print(f"  Output directory: {OUTPUT_DIR}")
    print()


# ─────────────────────────────────────────────
# MAIN — ties everything together
# ─────────────────────────────────────────────

def main():
    print()
    print("=" * 50)
    print("  MUTUAL FUND COHORT — SYNTHETIC DATA GENERATOR")
    print("=" * 50)
    print()

    make_output_dir()

    # Step 1: users
    users = build_users()
    write_users_csv(users)

    # Step 2: events
    events, investor_count, dormant_90_count = build_events(users)
    write_events_csv(events)

    # Step 3: summary
    print_summary(users, events, investor_count, dormant_90_count)


if __name__ == "__main__":
    main()