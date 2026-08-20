"""
backtest.py — offline historical backtest: for every date in the local
cached batted-ball history, scores every batter who actually played that
day using ONLY their point-in-time trailing power (data strictly before
that date) plus today's calibrated park/weather factors, then checks the
score against what actually happened.

WHY THIS IS FAST, NO LIVE API CALLS NEEDED: point-in-time correctness for
power_score comes for free from historical_features.add_batter_power's
closed='left' rolling window — it already computes a valid trailing value
for EVERY date in the dataset, not just "today." This script is really
just the first thing to actually USE those other dates' values instead of
discarding them the way hr_scanner_auto.py does. Venue, weather, and each
batter's real handedness for that at-bat are all already sitting in
data/batted_balls_*.parquet and data/weather_cache.csv from the season's
worth of daily_update.py runs — no schedule/roster API calls required.

SCOPE, DELIBERATELY:
  - Stage 1 only (power x park x weather) — no matchup layer. Matchup
    needs live per-batter Savant pulls; doing that for 100+ historical
    days would mean thousands of calls and real rate-limit risk. A
    separate, slower tool could add that later for a subset of dates.
  - Uses TODAY's calibrated park_factors_live.json / weather_factors_live.json
    for every historical date, not a walk-forward recalibration using only
    data prior to that date. Real simplification: today's calibration has
    some hindsight benefit from the whole season. Good enough to sanity-
    check whether the SCORING approach has real signal; not a fully
    rigorous "what would we have known in real time" test — treat the
    RELATIVE pattern (does higher score = higher real HR rate) as the
    trustworthy part, and the exact calibration-table numbers as directional.
  - A batter with fewer than MIN_PRIOR_BATTED_BALLS of trailing history as
    of a given date is skipped for that date rather than backfilled with
    a fake number — April/early May dates will have smaller (sometimes
    empty) slates until enough of the season has actually happened. That's
    expected, not a bug — same reasoning the live scanner already uses.
  - Batters with zero batted balls on a given date (all walks/strikeouts)
    aren't in the local cache at all, so they're invisible here. Doesn't
    affect HR-detection accuracy (they didn't hit one), just means the
    daily "field" is a bit smaller than the true full lineup.

RUN:
    python3 backtest.py                        # every backtestable date in the cache
    python3 backtest.py 2026-05-01 2026-06-01   # a specific date range
"""

import sys

import pandas as pd

from historical_features import load_batted_balls, add_batter_power, MIN_PRIOR_BATTED_BALLS
from hr_scanner import get_park_factor, weather_adjustment, Game
from calibrate_weather import categorize_wind

TRAILING_HR_RATE_ELITE = 0.13  # keep in sync with hr_scanner_auto.py's constant of the same name

BACKTEST_LOG_PATH = "data/backtest_log.csv"


def load_weather_cache() -> pd.DataFrame:
    try:
        return pd.read_csv("data/weather_cache.csv")
    except FileNotFoundError:
        print("No data/weather_cache.csv found — run historical_data.py first.")
        sys.exit(1)


def build_backtest_rows(start_dt: str | None, end_dt: str | None) -> pd.DataFrame:
    print("Loading cached batted-ball history...")
    batted = load_batted_balls(None)
    weather = load_weather_cache()

    print("Computing point-in-time trailing power for every date in the season "
          "(same computation calibrate_model.py already runs, just not discarded this time)...")
    with_power = add_batter_power(batted)
    with_power["is_hr"] = (with_power["events"] == "home_run").astype(int)

    # Actual outcome per (batter, game): did they hit at least one HR that game?
    outcomes = (
        with_power.groupby(["batter", "game_pk"])["is_hr"]
        .sum()
        .reset_index(name="hr_count")
    )
    outcomes["hit_hr"] = (outcomes["hr_count"] > 0).astype(int)

    # One row per (batter, game_pk) with their PRE-GAME trailing power. A batter can
    # have several batted balls in one game; they all share the same pre-game value
    # (closed='left' already excludes that day's own at-bats from the rolling window),
    # so "first" is fine here, not an average-across-at-bats situation.
    per_game = (
        with_power.dropna(subset=["game_date"])
        .sort_values(["batter", "game_pk"])
        .groupby(["batter", "game_pk"], as_index=False)
        .agg(game_date=("game_date", "first"), stand=("stand", "first"),
             batter_power=("batter_power", "first"))
    )
    per_game = per_game.merge(outcomes[["batter", "game_pk", "hit_hr", "hr_count"]],
                               on=["batter", "game_pk"], how="left")
    per_game = per_game.merge(weather[["game_pk", "venue", "temp", "condition", "wind"]],
                               on="game_pk", how="left")

    before = len(per_game)
    per_game = per_game.dropna(subset=["batter_power"])
    print(f"  {before - len(per_game)} batter-games dropped (fewer than "
          f"{MIN_PRIOR_BATTED_BALLS} trailing batted balls as of that date — "
          f"too early in that player's season, or too early in the season overall).")

    per_game["game_date"] = pd.to_datetime(per_game["game_date"]).dt.date.astype(str)
    if start_dt:
        per_game = per_game[per_game["game_date"] >= start_dt]
    if end_dt:
        per_game = per_game[per_game["game_date"] <= end_dt]
    return per_game


def score_rows(per_game: pd.DataFrame) -> pd.DataFrame:
    print(f"Scoring {len(per_game)} historical batter-games with today's calibrated "
          f"park/weather factors (Stage 1 only — no matchup layer)...")
    scores, park_vals, wx_vals, power_vals = [], [], [], []
    for _, row in per_game.iterrows():
        power = min(row["batter_power"] / TRAILING_HR_RATE_ELITE, 1.0)
        park = get_park_factor(row.get("venue") or "", row["stand"])

        is_dome = str(row.get("condition") or "").lower() in ("roof closed", "dome")
        speed, wind_dir = categorize_wind(row.get("wind") or "")
        g = Game(home_team="", away_team="", park=row.get("venue") or "", is_dome=is_dome,
                 wind_speed_mph=speed, wind_dir=wind_dir, temp=row.get("temp"))
        wx = weather_adjustment(g, row["stand"])

        power_vals.append(round(power, 3))
        park_vals.append(park)
        wx_vals.append(wx)
        scores.append(round(power * (1 + park) * (1 + wx), 4))

    out = per_game.copy()
    out["power_score"] = power_vals
    out["park_factor"] = park_vals
    out["weather_adj"] = wx_vals
    out["score"] = scores
    out["situational_boost"] = out.apply(
        lambda r: round(r["score"] / r["power_score"] - 1, 4) if r["power_score"] else 0.0, axis=1
    )
    out["rank"] = out.groupby("game_date")["score"].rank(ascending=False, method="first").astype(int)
    out["date"] = out["game_date"]
    out["resolved"] = True
    return out


if __name__ == "__main__":
    start_dt = sys.argv[1] if len(sys.argv) > 1 else None
    end_dt = sys.argv[2] if len(sys.argv) > 2 else None

    print("NOTE: this backtest applies TODAY's calibrated park/weather factors to every "
          "historical date — it's a fast check of whether the scoring approach has real "
          "signal, not a fully rigorous no-hindsight replay. See the module docstring.\n")

    per_game = build_backtest_rows(start_dt, end_dt)
    if per_game.empty:
        print("No backtestable batter-games found for this range.")
        sys.exit(0)

    scored = score_rows(per_game)
    scored.to_csv(BACKTEST_LOG_PATH, index=False)

    n_dates = scored["date"].nunique()
    print(f"\nBacktested {len(scored)} batter-games across {n_dates} dates.")
    print(f"Saved to {BACKTEST_LOG_PATH}")

    import check_results as cr
    cr.print_summary(scored)
