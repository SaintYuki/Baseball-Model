"""
check_results.py — resolves past predictions against real Statcast outcomes,
and reports on how the model's actually doing.

WHY THIS EXISTS: hr_scanner_auto.py logs every day's predicted slate to
data/predictions_log.csv (see log_predictions() there), but a prediction is
only useful once we know whether it came true. Without this, there's no way
to tell if the ranking has any real signal, whether the Value Board picks
are actually finding value, or what a given score number really means in
terms of real HR odds. This script:

  1. Finds any past dates in the log that are at least a day old (Statcast
     needs a day to finalize — same reasoning historical_data.py already
     uses for its own pulls) and haven't been resolved yet.
  2. Pulls real Statcast results for those dates and marks which predicted
     players actually hit a home run that day.
  3. Handles postponements/rainouts honestly: if a predicted game's game_pk
     never shows up at all in that date's real pitch data, those
     predictions are left VOIDED (unresolved), not silently scored as a
     miss — the game didn't happen, so there was never really a chance to
     check.
  4. Prints a performance summary: hit rate by rank tier, hit rate for
     situational-boost ("Value Board") plays vs. the rest, and — once
     there's enough resolved history — a rough score-to-real-HR-rate
     calibration table, which is the actual answer to "what does a score
     of 0.6 really mean."

RUN:
    python3 check_results.py                 # resolve what's resolvable, then report
    python3 check_results.py --summary-only   # skip resolving, just report on what's already resolved
"""

import sys
from datetime import date, timedelta

import pandas as pd
import pybaseball as pyb
import pybaseball.cache as pyb_cache

pyb_cache.enable()

PREDICTIONS_LOG_PATH = "data/predictions_log.csv"
MIN_RESOLVED_FOR_SUMMARY = 30  # below this, hit-rate breakdowns are too noisy to mean much


def load_log() -> pd.DataFrame:
    try:
        return pd.read_csv(PREDICTIONS_LOG_PATH)
    except FileNotFoundError:
        print(f"No {PREDICTIONS_LOG_PATH} found yet — run hr_scanner_auto.py at least once first; "
              f"it logs predictions automatically.")
        sys.exit(1)


def resolve_dates(log: pd.DataFrame) -> pd.DataFrame:
    log = log.copy()
    log["resolved"] = log["resolved"].astype(bool)
    yesterday = date.today() - timedelta(days=1)

    unresolved_dates = sorted(
        d for d in log.loc[~log["resolved"], "date"].unique()
        if date.fromisoformat(d) <= yesterday
    )
    if not unresolved_dates:
        print("Nothing new to resolve — every past date in the log is already checked.")
        return log

    start_dt, end_dt = unresolved_dates[0], unresolved_dates[-1]
    print(f"Resolving {len(unresolved_dates)} date(s): {start_dt} to {end_dt}...")
    real = pyb.statcast(start_dt, end_dt)
    if real.empty:
        print("  No Statcast data returned for this range — too recent to be finalized, or a "
              "bad range. Try again later.")
        return log

    hr_events = real[real["events"] == "home_run"]
    hr_counts = hr_events.groupby("batter").size().to_dict()
    games_with_data = set(real["game_pk"].dropna().astype(int))

    newly_resolved = 0
    for d in unresolved_dates:
        mask = (log["date"] == d) & (~log["resolved"])
        for idx in log[mask].index:
            row = log.loc[idx]
            gpk = row.get("game_pk")
            if pd.isna(gpk) or int(gpk) not in games_with_data:
                # This game never shows up at all in the real day's data — most likely
                # postponed or rained out. Leave it unresolved rather than wrongly
                # scoring it as a miss; there was never a real chance to check.
                continue
            bid = row.get("mlbam_id")
            hr_count = hr_counts.get(int(bid), 0) if pd.notna(bid) else 0
            log.loc[idx, "hr_count"] = hr_count
            log.loc[idx, "hit_hr"] = 1 if hr_count > 0 else 0
            log.loc[idx, "resolved"] = True
            newly_resolved += 1

    log.to_csv(PREDICTIONS_LOG_PATH, index=False)
    print(f"  Resolved {newly_resolved} predictions.")

    still_open = log[log["date"].isin(unresolved_dates) & ~log["resolved"]]
    if len(still_open):
        voided_games = sorted(still_open["game"].unique().tolist())
        print(f"  {len(still_open)} predictions still unresolved (game_pk never appeared in the "
              f"real data — likely postponed/rained out, will retry next run): {voided_games}")
    return log


def print_summary(log: pd.DataFrame):
    resolved = log[log["resolved"] == True].copy()  # noqa: E712 (explicit True match, not truthiness)
    n = len(resolved)
    print()
    print("=" * 78)
    print(f"MODEL PERFORMANCE SUMMARY ({n} resolved predictions across "
          f"{resolved['date'].nunique() if n else 0} day(s))")
    print("=" * 78)

    if n < MIN_RESOLVED_FOR_SUMMARY:
        print(f"Only {n} resolved predictions so far — need at least {MIN_RESOLVED_FOR_SUMMARY} "
              f"before hit-rate breakdowns mean much of anything. Keep running hr_scanner_auto.py "
              f"daily and check_results.py after games finish; check back in a week or two.")
        return

    def hit_rate(df):
        return df["hit_hr"].mean() if len(df) else float("nan")

    top10 = resolved[resolved["rank"] <= 10]
    top11_50 = resolved[(resolved["rank"] > 10) & (resolved["rank"] <= 50)]
    rest = resolved[resolved["rank"] > 50]
    overall = hit_rate(resolved)

    print(f"\nOverall HR rate across all logged predictions: {overall*100:.1f}%  (n={n})")
    print(f"  Rank 1-10:   {hit_rate(top10)*100:.1f}%  (n={len(top10)})")
    print(f"  Rank 11-50:  {hit_rate(top11_50)*100:.1f}%  (n={len(top11_50)})")
    print(f"  Rank 51+:    {hit_rate(rest)*100:.1f}%  (n={len(rest)})")
    print("  -> if the ranking has real signal, rank 1-10 should clear rank 51+ by a real margin.")

    boosted = resolved[resolved["situational_boost"] > 0.05]
    unboosted = resolved[resolved["situational_boost"] <= 0.05]
    print(f"\nSituational boost >5% (Value-Board-style plays): {hit_rate(boosted)*100:.1f}%  "
          f"(n={len(boosted)})")
    print(f"Situational boost <=5%:                            {hit_rate(unboosted)*100:.1f}%  "
          f"(n={len(unboosted)})")
    print("  -> this is the actual test of whether situational-lift plays are worth anything, "
          "beyond just looking clever.")

    print(f"\nScore -> real HR rate (rough calibration, quintiles):")
    try:
        resolved["score_bucket"] = pd.qcut(resolved["score"], 5, duplicates="drop")
        calib = resolved.groupby("score_bucket", observed=True)["hit_hr"].agg(["mean", "count"])
        for bucket, row in calib.iterrows():
            print(f"  score {bucket}:  actual HR rate {row['mean']*100:.1f}%  (n={int(row['count'])})")
        print("  -> THIS is what a given score number actually means in real terms — read the score "
              "off this table, not as a literal percentage.")
    except ValueError:
        print("  Not enough score spread yet to bucket meaningfully.")


if __name__ == "__main__":
    log = load_log()
    if "--summary-only" not in sys.argv:
        log = resolve_dates(log)
    print_summary(log)
