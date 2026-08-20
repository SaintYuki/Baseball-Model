"""
calibrate_weather.py — Phase 3: real empirical weather effects on HR rate,
bucketed by actual recorded temperature + wind (not a rolling time window —
see the Aug 8 discussion: a 75F calm day in April and a 75F calm day in
August are the same physical bucket, so pooling the full season by actual
conditions gives more data without the seasonal confound a pure recency
window would still carry).

WHAT IT DOES
  1. Loads batted-ball parquet(s) + weather_cache.csv from historical_data.py
  2. Drops dome games — a closed roof isn't "calm weather," it's not weather
     at all, and including it would contaminate the true calm-outdoor bucket
  3. Parses wind into (speed_mph, direction category: Calm/Out/In/Cross)
  4. Buckets temp (<60 / 60-70 / 70-80 / 80-90 / 90+) and wind speed
     (0-5 / 5-10 / 10-15 / 15+)
  5. Computes HR rate per (temp_bucket, wind_dir, wind_speed_bucket) vs the
     outdoor league average, same shrinkage + confidence-flag treatment as
     calibrate_park_factors.py

RUN:
    python3 calibrate_weather.py
    python3 calibrate_weather.py data/batted_balls_2025-04-01_2025-09-30.parquet
"""

import glob
import sys

import pandas as pd

MIN_SAMPLE = 300
SHRINKAGE_K = 300

DOME_CONDITIONS = {"roof closed", "dome"}


def load_batted_balls(path_arg: str | None) -> pd.DataFrame:
    if path_arg:
        return pd.read_parquet(path_arg)
    files = glob.glob("data/batted_balls_*.parquet")
    if not files:
        print("No cached batted-ball files found in ./data/ — run historical_data.py first.")
        sys.exit(1)
    print(f"Loading {len(files)} cached file(s): {files}")
    combined = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # date ranges pulled at different times can overlap (e.g. an early pilot pull whose
    # range is fully contained in a later full-season pull) — glob-and-concat alone
    # double-counts every pitch in the overlap. game_pk+at_bat_number+pitch_number
    # uniquely identifies a single pitch in Statcast data, so dedup on that.
    before = len(combined)
    combined = combined.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
    if before != len(combined):
        print(f"Removed {before - len(combined)} duplicate rows from overlapping cached files.")
    return combined


def load_weather() -> pd.DataFrame:
    try:
        return pd.read_csv("data/weather_cache.csv")
    except FileNotFoundError:
        print("No weather_cache.csv found — run historical_data.py first.")
        sys.exit(1)


def categorize_wind(wind_str) -> tuple[float, str]:
    """Returns (speed_mph, direction_category). Category is one of
    Calm / Out / In / Cross / Varies / Unknown."""
    s = str(wind_str).strip().lower()
    if not s or s == "nan":
        return 0.0, "Calm"
    try:
        speed = float(s.split("mph")[0].strip())
    except ValueError:
        speed = 0.0
    if "out" in s:
        return speed, "Out"
    if "in from" in s:
        return speed, "In"
    if "to r" in s or "to l" in s:
        return speed, "Cross"
    if "calm" in s or "none" in s:
        return speed, "Calm"
    if "varies" in s:
        return speed, "Varies"
    return speed, "Unknown"


def bucket_temp(temp) -> str:
    try:
        t = float(temp)
    except (ValueError, TypeError):
        return "Unknown"
    if t < 60:
        return "<60"
    if t < 70:
        return "60-70"
    if t < 80:
        return "70-80"
    if t < 90:
        return "80-90"
    return "90+"


def bucket_speed(speed: float) -> str:
    if speed < 5:
        return "0-5"
    if speed < 10:
        return "5-10"
    if speed < 15:
        return "10-15"
    return "15+"


def compute_weather_factors(batted: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = batted.merge(weather, on="game_pk", how="left")
    df["is_hr"] = (df["events"] == "home_run").astype(int)

    df["condition_lower"] = df["condition"].astype(str).str.lower()
    before = len(df)
    df = df[~df["condition_lower"].isin(DOME_CONDITIONS)]
    print(f"Dropped {before - len(df)} dome-game batted balls, {len(df)} outdoor batted balls remain.")

    df[["wind_speed", "wind_dir"]] = df["wind"].apply(lambda w: pd.Series(categorize_wind(w)))
    df["temp_bucket"] = df["temp"].apply(bucket_temp)
    df["speed_bucket"] = df["wind_speed"].apply(bucket_speed)

    unknown_wind = (df["wind_dir"] == "Unknown").sum()
    if unknown_wind:
        print(f"NOTE: {unknown_wind} rows had an unrecognized wind format — check df['wind'] values if this is a big chunk.")

    league_avg = df["is_hr"].mean()
    print(f"\nOutdoor league-wide HR rate per batted ball: {league_avg:.4f}  (n={len(df)})")

    grouped = (
        df.groupby(["temp_bucket", "wind_dir", "speed_bucket"])
        .agg(batted_balls=("is_hr", "count"), hr_rate=("is_hr", "mean"))
        .reset_index()
    )
    grouped["weather_factor_raw"] = grouped["hr_rate"] / league_avg - 1
    grouped["shrunk_rate"] = (
        (grouped["batted_balls"] * grouped["hr_rate"] + SHRINKAGE_K * league_avg)
        / (grouped["batted_balls"] + SHRINKAGE_K)
    )
    grouped["weather_factor"] = grouped["shrunk_rate"] / league_avg - 1
    grouped["confidence"] = grouped["batted_balls"].apply(
        lambda n: "OK" if n >= MIN_SAMPLE else f"LOW CONF (n={n}, want {MIN_SAMPLE}+)"
    )
    return grouped.sort_values("weather_factor", ascending=False)


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    batted = load_batted_balls(path_arg)
    weather = load_weather()

    result = compute_weather_factors(batted, weather)

    print("\n" + "=" * 88)
    print("EMPIRICAL WEATHER FACTORS (sorted best -> worst for hitters)")
    print("=" * 88)
    for _, row in result.iterrows():
        print(f"  temp {row['temp_bucket']:<7} wind {row['wind_dir']:<8} {row['speed_bucket']:<6}mph  "
              f"factor={row['weather_factor']:+.3f}  (raw {row['weather_factor_raw']:+.3f})  "
              f"hr_rate={row['hr_rate']:.4f}  {row['confidence']}")

    out_path = "data/weather_factors_empirical.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved full table to {out_path}")

    low_conf = (result["confidence"] != "OK").sum()
    if low_conf:
        print(f"\n{low_conf} of {len(result)} buckets are LOW CONF. Sparse cells (e.g. 15+ mph tailwinds) "
              f"are just rare events — more seasons of data is the only real fix for those specifically, "
              f"the common buckets (Calm, 5-10mph) should already look solid off one season.")
