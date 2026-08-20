"""
walk_forward_backtest.py — the rigorous version of backtest.py. That tool
applies TODAY's park/weather calibration (which has hindsight from the
whole season) to every historical date. This one instead recalibrates
periodically using an EXPANDING window of ONLY data strictly before each
window, and applies that calibration only to the dates that follow it, then
moves forward and recalibrates again. No date's score is ever informed by
data from after that date — the leakage backtest.py's own docstring flagged
is fully removed here.

MUCH SLOWER THAN backtest.py, ON PURPOSE: each recalibration step is a real
run of calibrate_model.py's own pipeline — full pitch-level pull, trailing
matchup features, two logistic regression fits (reuses those exact
functions, not a reimplementation). Doing this every
RECALIBRATION_INTERVAL_DAYS (default: 14) across a season means running
that full pipeline maybe 8-12 times, not once. This is a "kick it off and
let it run" tool, not a quick check.

WHAT STAYS THE SAME AS backtest.py: batter_power (trailing form) is already
point-in-time correct by construction (historical_features.add_batter_power's
closed='left' rolling window) — computed once across the full history and
read off per-row, same as before. Only the PARK/WEATHER calibration piece
needed fixing, since that's the part that was using hindsight.

EXPECT FEW OR ZERO SIGNIFICANT FACTORS IN THE EARLY WINDOWS — the first
recalibration only has ~35 days of season to work with, versus the ~4-5
months the current live calibration has. That's not a bug, it's this tool
doing its job: showing what would genuinely have been knowable that early
in a real season, not what hindsight now reveals.

RUN:
    python3 walk_forward_backtest.py
    python3 walk_forward_backtest.py 2026-05-15 2026-08-01   # limit the test range
"""

import re
import sys
from datetime import date, timedelta

import pandas as pd

from historical_features import add_batter_power
from calibrate_model import load_batted_balls, load_weather, build_dataset, fit_model, extract_park_factors, extract_weather_factors
from calibrate_weather import categorize_wind, bucket_temp, bucket_speed

TRAILING_HR_RATE_ELITE = 0.13     # keep in sync with hr_scanner_auto.py's constant of the same name
SITUATIONAL_FACTOR_CLAMP = 0.25   # keep in sync with hr_scanner.py's constant of the same name

RECALIBRATION_INTERVAL_DAYS = 14  # how often to refit park/weather using only prior data
MIN_CALIBRATION_DAYS = 35         # need at least this many days of history before the first fit

BACKTEST_LOG_PATH = "data/walk_forward_backtest_log.csv"

WEATHER_KEY_RE = re.compile(r"temp=(\S+) wind=(\S+) speed=(\S+)mph")


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def factors_from_model_output(park_df: pd.DataFrame, weather_df: pd.DataFrame) -> tuple[dict, dict]:
    """Same significance rule calibrate_model.py's own output already
    encodes (a factor's 'significant' column) — not the stability-gating
    from export_live_factors.py, which is a separate, later safeguard for
    the LIVE pipeline against noisy single-day snapshots. This tool's whole
    point is removing hindsight leakage, a different concern; each window's
    fit here already uses months of prior data, not a single day."""
    park_factors: dict = {}
    for _, row in park_df.iterrows():
        park_factors.setdefault(row["key"], {})
        park_factors[row["key"]][row["stand"]] = (
            round(float(row["factor"]), 4) if row["significant"] == "yes" else 0.0
        )

    weather_factors: dict = {"L": {}, "R": {}}
    for _, row in weather_df.iterrows():
        m = WEATHER_KEY_RE.match(row["key"])
        if not m:
            continue
        temp_bucket, wind_dir, speed_bucket = m.groups()
        stand = row["stand"]
        weather_factors[stand].setdefault(temp_bucket, {}).setdefault(wind_dir, {})
        weather_factors[stand][temp_bucket][wind_dir][speed_bucket] = (
            round(float(row["factor"]), 4) if row["significant"] == "yes" else 0.0
        )
    return park_factors, weather_factors


def get_park_factor(park_factors: dict, park: str, stand: str) -> float:
    raw = park_factors.get(park, {}).get(stand, 0.0)
    return _clamp(raw, SITUATIONAL_FACTOR_CLAMP)


def get_weather_factor(weather_factors: dict, stand: str, temp, wind_dir: str, speed: float, is_dome: bool) -> float:
    if is_dome or temp is None or not wind_dir:
        return 0.0
    raw = (
        weather_factors.get(stand, {})
        .get(bucket_temp(temp), {})
        .get(wind_dir, {})
        .get(bucket_speed(speed), 0.0)
    )
    return _clamp(raw, SITUATIONAL_FACTOR_CLAMP)


def prepare_per_game(all_batted: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """The scoring-side data: one row per (batter, game_pk) with each
    batter's point-in-time trailing power as of that game, the real
    outcome, and venue/weather. Identical in spirit to backtest.py's
    version — this piece never had leakage, so it doesn't change here."""
    with_power = add_batter_power(all_batted)
    with_power["is_hr"] = (with_power["events"] == "home_run").astype(int)
    outcomes = (
        with_power.groupby(["batter", "game_pk"])["is_hr"]
        .sum().reset_index(name="hr_count")
    )
    outcomes["hit_hr"] = (outcomes["hr_count"] > 0).astype(int)

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
    per_game = per_game.dropna(subset=["batter_power"])
    per_game["game_date"] = pd.to_datetime(per_game["game_date"]).dt.date
    return per_game


def recalibrate(all_batted: pd.DataFrame, weather: pd.DataFrame, cutoff: date) -> tuple[dict, dict, int, int]:
    """Fits park/weather factors using ONLY batted balls strictly before
    `cutoff`. Reuses calibrate_model.py's real build_dataset/fit_model/
    extract_* functions unchanged — not a reimplementation, so this is
    the exact same methodology the live calibration uses, just windowed."""
    train = all_batted[pd.to_datetime(all_batted["game_date"]).dt.date < cutoff].copy()
    if train.empty:
        return {}, {}, 0, 0

    df = build_dataset(train, weather)
    park_results, weather_results = [], []
    for hand in ["L", "R"]:
        (model, sub, avg_offense, avg_pitching, avg_power, avg_batter_rv,
         avg_pitcher_rv, ref_temp, ref_wind, ref_speed) = fit_model(df, hand)
        park_results.append(extract_park_factors(
            model, sub, hand, avg_offense, avg_pitching, avg_power,
            avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed))
        weather_results.append(extract_weather_factors(
            model, sub, hand, avg_offense, avg_pitching, avg_power, avg_batter_rv, avg_pitcher_rv))
    park_df = pd.concat(park_results, ignore_index=True)
    weather_df = pd.concat(weather_results, ignore_index=True)

    park_factors, weather_factors = factors_from_model_output(park_df, weather_df)
    n_park_sig = int((park_df["significant"] == "yes").sum())
    n_wx_sig = int((weather_df["significant"] == "yes").sum())
    return park_factors, weather_factors, n_park_sig, n_wx_sig


def score_window(window_rows: pd.DataFrame, park_factors: dict, weather_factors: dict) -> list[dict]:
    scored = []
    for _, row in window_rows.iterrows():
        power = min(row["batter_power"] / TRAILING_HR_RATE_ELITE, 1.0)
        park = get_park_factor(park_factors, row.get("venue") or "", row["stand"])
        speed, wind_dir = categorize_wind(row.get("wind") or "")
        is_dome = str(row.get("condition") or "").lower() in ("roof closed", "dome")
        wx = get_weather_factor(weather_factors, row["stand"], row.get("temp"), wind_dir, speed, is_dome)
        score = round(power * (1 + park) * (1 + wx), 4)
        scored.append({
            "date": row["game_date"].isoformat(), "batter": row["batter"], "game_pk": row["game_pk"],
            "power_score": round(power, 3), "park_factor": park, "weather_adj": wx, "score": score,
            "situational_boost": round(score / power - 1, 4) if power else 0.0,
            "hit_hr": row["hit_hr"], "hr_count": row["hr_count"],
        })
    return scored


def run(all_batted: pd.DataFrame, weather: pd.DataFrame, per_game: pd.DataFrame,
        test_start: date, test_end: date) -> pd.DataFrame:
    scored_chunks = []
    cutoff = test_start
    step = 0
    while cutoff <= test_end:
        window_end = min(cutoff + timedelta(days=RECALIBRATION_INTERVAL_DAYS), test_end + timedelta(days=1))
        step += 1
        print(f"--- Window {step}: recalibrating on data before {cutoff}, "
              f"scoring {cutoff} through {window_end - timedelta(days=1)} ---")
        try:
            park_factors, weather_factors, n_park_sig, n_wx_sig = recalibrate(all_batted, weather, cutoff)
            print(f"  {n_park_sig} significant park factors, {n_wx_sig} significant weather buckets "
                  f"(using only data before {cutoff}).")
        except Exception as e:
            print(f"  Recalibration failed for this window ({e}) — scoring this window at fully "
                  f"neutral park/weather rather than guessing.")
            park_factors, weather_factors = {}, {}

        window_rows = per_game[(per_game["game_date"] >= cutoff) & (per_game["game_date"] < window_end)]
        scored_chunks.extend(score_window(window_rows, park_factors, weather_factors))
        cutoff = window_end

    if not scored_chunks:
        return pd.DataFrame()
    scored = pd.DataFrame(scored_chunks)
    scored["rank"] = scored.groupby("date")["score"].rank(ascending=False, method="first").astype(int)
    scored["resolved"] = True
    return scored


if __name__ == "__main__":
    start_arg = sys.argv[1] if len(sys.argv) > 1 else None
    end_arg = sys.argv[2] if len(sys.argv) > 2 else None

    print("NOTE: this recalibrates park/weather factors periodically using only data strictly "
          "before each window, then applies each fit only to the dates that follow it. No date's "
          "score is ever informed by data from after that date. This is much slower than "
          "backtest.py by design — see the module docstring.\n")

    print("Loading cached batted-ball + weather history...")
    all_batted = load_batted_balls(None)
    weather = load_weather()

    print("Computing point-in-time trailing power for every date "
          "(unchanged from backtest.py — this piece never had leakage)...")
    per_game = prepare_per_game(all_batted, weather)

    all_dates = sorted(per_game["game_date"].unique())
    if not all_dates:
        print("No backtestable batter-games found at all.")
        sys.exit(0)
    min_date, max_date = all_dates[0], all_dates[-1]

    test_start = date.fromisoformat(start_arg) if start_arg else min_date + timedelta(days=MIN_CALIBRATION_DAYS)
    test_end = date.fromisoformat(end_arg) if end_arg else max_date

    if test_start > test_end:
        print(f"Not enough history yet — need at least {MIN_CALIBRATION_DAYS} days before the "
              f"first recalibration, but the cached data only spans {min_date} to {max_date}.")
        sys.exit(0)

    print(f"Walk-forward backtest from {test_start} to {test_end}, "
          f"recalibrating every {RECALIBRATION_INTERVAL_DAYS} days.\n")

    scored = run(all_batted, weather, per_game, test_start, test_end)
    if scored.empty:
        print("No dates were backtested for this range.")
        sys.exit(0)

    scored.to_csv(BACKTEST_LOG_PATH, index=False)
    print(f"\nWalk-forward backtested {len(scored)} batter-games across {scored['date'].nunique()} dates.")
    print(f"Saved to {BACKTEST_LOG_PATH}")

    import check_results as cr
    cr.print_summary(scored)
