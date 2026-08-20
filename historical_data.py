"""
historical_data.py — Phase 1 of the calibration project: pull real historical
batted-ball outcomes + real per-game weather, so Phases 2-4 (empirical park
factors, empirical weather effects, feature-weight modeling) have something
real to run against instead of the hardcoded approximations in hr_scanner.py.

WHY START SMALL: every data source we've touched so far had a surprise in it
(the name column format, the empty live-weather field, the barrel-rate
column). Default here is a 14-day pilot — enough to validate the pipeline
and eyeball real numbers before committing to a 20-30+ minute full-season
weather pull. Expand the date range once the pilot output looks sane.

WHAT THIS PULLS
  1. Every batted ball (ball put in play) league-wide for the date range,
     via pybaseball.statcast() — includes game_pk, park (home_team as proxy
     until we map to venue name), batter/pitcher id, stand, p_throws,
     launch_speed, launch_angle, bb_type, and events (tells us if it was a HR).
  2. Historical weather for every unique game in that batted-ball set, via
     the same MLB Stats API game-feed endpoint we already use for tonight's
     live weather — the open question we're testing here is whether that
     field is still populated on COMPLETED historical games, since it only
     works right before/during live games for tonight's slate. If the pilot
     comes back with a low hit-rate on weather, that tells us this endpoint
     doesn't retain historical weather and we need a different source
     (e.g. an external weather API keyed to park lat/long + game date/time).

OUTPUT: two local cache files in ./data/
  - batted_balls_<start>_<end>.parquet
  - weather_cache.csv (append-only, keyed by game_pk — safe to re-run,
    already-fetched games are skipped)

SETUP: pip install pyarrow   (parquet support, if not already installed)
RUN:
    python3 historical_data.py                          # 14-day pilot, defaults
    python3 historical_data.py 2025-04-01 2025-09-30     # full season, once validated
"""

import os
import sys
import time
from datetime import date, timedelta

import pandas as pd
import statsapi
import pybaseball as pyb
import pybaseball.cache as pyb_cache

pyb_cache.enable()  # statcast() calls get cached to disk automatically, ~1yr expiry

DATA_DIR = "data"
WEATHER_CACHE_PATH = os.path.join(DATA_DIR, "weather_cache.csv")
REQUEST_DELAY_SEC = 0.3


def pull_batted_balls(start_dt: str, end_dt: str) -> pd.DataFrame:
    """League-wide balls put in play for the date range. type == 'X' is
    Statcast's code for 'ball put in play' — filters out pitches that didn't
    result in contact, since that's what park/weather physically act on."""
    print(f"Pulling league-wide Statcast data {start_dt} to {end_dt} (cached after first pull)...")
    df = pyb.statcast(start_dt, end_dt)
    if df.empty:
        print("  No data returned — check the date range (no games in range?).")
        return df
    batted = df[df["type"] == "X"].copy()
    print(f"  {len(df)} total pitches -> {len(batted)} balls put in play.")
    return batted


def load_weather_cache() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(WEATHER_CACHE_PATH):
        return pd.read_csv(WEATHER_CACHE_PATH)
    return pd.DataFrame(columns=["game_pk", "temp", "condition", "wind", "venue"])


def pull_historical_weather(game_pks: list) -> pd.DataFrame:
    """Pulls gameData.weather for each game_pk, skipping ones already cached.
    Appends to the CSV as it goes, so a Ctrl+C mid-pull doesn't lose progress —
    just rerun and it picks up where it left off."""
    cached = load_weather_cache()
    already_have = set(cached["game_pk"].astype(int)) if not cached.empty else set()
    to_fetch = [pk for pk in game_pks if int(pk) not in already_have]

    print(f"\n{len(already_have)} games already cached, {len(to_fetch)} left to fetch.")
    if not to_fetch:
        return cached

    new_rows = []
    for i, pk in enumerate(to_fetch, 1):
        try:
            feed = statsapi.get("game", {"gamePk": int(pk)})
            wx = feed.get("gameData", {}).get("weather", {}) or {}
            venue = feed.get("gameData", {}).get("venue", {}).get("name", "")
            new_rows.append({
                "game_pk": int(pk),
                "temp": wx.get("temp", ""),
                "condition": wx.get("condition", ""),
                "wind": wx.get("wind", ""),
                "venue": venue,
            })
        except Exception as e:
            new_rows.append({"game_pk": int(pk), "temp": "", "condition": "", "wind": "", "venue": f"ERROR: {e}"})

        if i % 25 == 0 or i == len(to_fetch):
            print(f"  [{i}/{len(to_fetch)}] fetched...")
            # flush progress to disk periodically, not just at the very end
            pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True).to_csv(WEATHER_CACHE_PATH, index=False)

        time.sleep(REQUEST_DELAY_SEC)

    full = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
    full.to_csv(WEATHER_CACHE_PATH, index=False)
    return full


if __name__ == "__main__":
    if len(sys.argv) == 3:
        start_dt, end_dt = sys.argv[1], sys.argv[2]
    else:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=14)
        start_dt, end_dt = start.isoformat(), end.isoformat()
        print(f"No date range given — running a 14-day pilot: {start_dt} to {end_dt}\n")

    os.makedirs(DATA_DIR, exist_ok=True)

    batted = pull_batted_balls(start_dt, end_dt)
    if batted.empty:
        sys.exit(1)

    out_path = os.path.join(DATA_DIR, f"batted_balls_{start_dt}_{end_dt}.parquet")
    batted.to_parquet(out_path, index=False)
    print(f"  Saved to {out_path}")

    unique_games = batted["game_pk"].dropna().unique().tolist()
    weather = pull_historical_weather(unique_games)

    # VALIDATION REPORT — this is the part that actually answers our open question
    matched = weather[weather["game_pk"].isin([int(g) for g in unique_games])]
    hit_rate = (matched["temp"].astype(str).str.strip() != "").mean() if not matched.empty else 0

    print("\n" + "=" * 70)
    print("PILOT VALIDATION REPORT")
    print("=" * 70)
    print(f"Batted balls pulled:      {len(batted)}")
    print(f"Unique games:             {len(unique_games)}")
    print(f"Games with weather data:  {int(hit_rate * len(matched))} / {len(matched)}  ({hit_rate*100:.0f}%)")
    if hit_rate > 0.8:
        print("-> Historical weather looks reliably available. Safe to scale up the date range.")
    elif hit_rate > 0.2:
        print("-> Partial coverage — some games have it, some don't. Worth digging into which/why before scaling up.")
    else:
        print("-> Historical weather is NOT reliably available from this endpoint. We'll need a different")
        print("   source (e.g. an external weather API by park lat/long + game date) before Phase 3 can work.")
    print("\nSample weather rows:")
    print(matched.head(5).to_string(index=False))

    # RENAME-FRAGMENTATION CHECK — catches a park that changed names mid-range,
    # which would otherwise silently split one park's sample across two venue strings.
    home_venue = batted[["home_team", "game_pk"]].drop_duplicates().merge(
        matched[["game_pk", "venue"]], on="game_pk", how="left"
    )
    venue_counts = home_venue.groupby("home_team")["venue"].nunique()
    renamed = venue_counts[venue_counts > 1]
    print("\n" + "=" * 70)
    print("VENUE-RENAME CHECK")
    print("=" * 70)
    if renamed.empty:
        print("Every team's home games used a single consistent venue name. Clean.")
    else:
        print("These teams show MULTIPLE venue names in this date range — likely a mid-range rename,")
        print("which will silently fragment that park's sample if not merged before analysis:")
        for team in renamed.index:
            names = home_venue[home_venue["home_team"] == team]["venue"].dropna().unique().tolist()
            print(f"  {team}: {names}")
