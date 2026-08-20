"""
export_live_factors.py — turns calibrate_model.py's output into lookup
tables hr_scanner.py can load at runtime, replacing the hardcoded PARK_FACTORS
guesses and the crude wind-speed-only weather heuristic with real, calibrated,
significance-gated, day-over-day STABLE numbers.

THE SIGNIFICANCE RULE: every park and every weather bucket that did NOT reach
statistical significance in today's model gets 0.0 (neutral), not its noisy
point estimate. The model's own honest conclusion for those was "not
distinguishable from a neutral park/condition" — using the point estimate
anyway would quietly contradict that finding just because a number is
sitting right there.

THE STABILITY RULE (new): significance on a single day isn't enough to trust
on its own — individual-day CIs are often wide even when they clear the
bar. Example that motivated this: Tropicana Field R showed +79% one day,
but its own CI spanned +12% to +181% — that's not a number to build a
day's picks around off one calibration run. So every day's significance
result per key gets appended to a running log
(data/park_stability_log.csv, data/weather_stability_log.csv), and a
factor is only exported as "real" once it's been significant for
MIN_CONSECUTIVE_DAYS *consecutive* calibration runs — and even then, the
exported value is the AVERAGE across that qualifying streak (capped to the
most recent MAX_AVERAGE_WINDOW days so a long streak doesn't go stale),
not just today's single noisier point estimate. If today's result isn't
significant, the streak resets immediately, even if it had a long run
before — a fresh non-significant day should cost trust right away, not
coast on stale history.

TRANSITION COST, WORTH KNOWING: the stability log starts empty. For the
first MIN_CONSECUTIVE_DAYS-1 days after adding this, EVERY factor exports
as neutral, even ones already well-established across many prior manual
runs (Wrigley Field L, etc.) — there's no way to backfill genuine
day-by-day history that was never logged. One-time cost, not ongoing;
lower MIN_CONSECUTIVE_DAYS if that bootstrap period matters.

OUTPUT:
    data/park_factors_live.json      {venue: {"L": factor, "R": factor}}
    data/weather_factors_live.json   {"L": {temp: {wind_dir: {speed: factor}}}, "R": {...}}
    data/park_stability_log.csv      running per-day significance history (park)
    data/weather_stability_log.csv   running per-day significance history (weather)

RUN (after calibrate_model.py has produced its CSVs):
    python3 export_live_factors.py
"""

import json
import re
from datetime import date

import pandas as pd

WEATHER_KEY_RE = re.compile(r"temp=(\S+) wind=(\S+) speed=(\S+)mph")

MIN_CONSECUTIVE_DAYS = 3   # a key must be significant this many days IN A ROW before it's trusted
MAX_AVERAGE_WINDOW = 14    # once qualified, average over at most this many most-recent days

PARK_STABILITY_LOG = "data/park_stability_log.csv"
WEATHER_STABILITY_LOG = "data/weather_stability_log.csv"


def _update_stability_log(log_path: str, today_rows: list[dict], today: str) -> pd.DataFrame:
    """Appends today's (key, stand, significant, factor) results to the
    running log. Replaces any existing rows for today's date first, so
    re-running export_live_factors.py twice in one day doesn't duplicate
    an entry or artificially pad a streak."""
    new_df = pd.DataFrame(today_rows)
    new_df["date"] = today
    try:
        existing = pd.read_csv(log_path)
        existing = existing[existing["date"].astype(str) != str(today)]
        combined = pd.concat([existing, new_df], ignore_index=True)
    except FileNotFoundError:
        combined = new_df
    combined.to_csv(log_path, index=False)
    return combined


def _qualifying_average(log: pd.DataFrame, key: str, stand: str, min_days: int,
                         max_average_window: int = MAX_AVERAGE_WINDOW):
    """Returns (average_factor, streak_length). average_factor is None
    unless this (key, stand) has been significant for the most recent
    `min_days` consecutive LOGGED entries — a gap (a skipped run) just
    doesn't add an entry, it doesn't unfairly break or pad a streak.
    Walks backward from the most recent entry; stops at the first
    non-significant one, so today's result always governs first."""
    rows = log[(log["key"] == key) & (log["stand"] == stand)].sort_values("date")
    if rows.empty:
        return None, 0
    sig = rows["significant"].tolist()
    factors = rows["factor"].tolist()
    streak = 0
    streak_factors = []
    for is_sig, factor in zip(reversed(sig), reversed(factors)):
        if is_sig != "yes":
            break
        streak += 1
        streak_factors.append(factor)
    if streak >= min_days:
        window = streak_factors[:max_average_window]  # most-recent-first already
        return sum(window) / len(window), streak
    return None, streak


def _print_almost_qualifying(log: pd.DataFrame, min_days: int):
    if min_days <= 1:
        return
    pairs = log[["key", "stand"]].drop_duplicates()
    almost = []
    for _, row in pairs.iterrows():
        avg, streak = _qualifying_average(log, row["key"], row["stand"], min_days)
        if avg is None and streak == min_days - 1:
            almost.append((row["key"], row["stand"], streak))
    if almost:
        print(f"  {len(almost)} key(s) one significant day away from qualifying:")
        for key, stand, streak in almost[:10]:
            print(f"    {key} ({stand}) — {streak}/{min_days} days")


def export_park_factors(path: str = "data/park_factors_model.csv", today: str | None = None,
                         stability_log_path: str = PARK_STABILITY_LOG, label: str = "Park factors",
                         min_consecutive_days: int = MIN_CONSECUTIVE_DAYS,
                         max_average_window: int = MAX_AVERAGE_WINDOW) -> dict:
    """stability_log_path/label let a different prop's calibration (e.g. the
    hits pilot in export_hits_live_factors.py) reuse this exact function
    with its own separate stability history, instead of accidentally
    writing into — and corrupting — the HR pipeline's log. Defaults match
    the original hardcoded behavior exactly, so the existing daily HR call
    (export_park_factors(today=today)) is completely unaffected."""
    today = today or date.today().isoformat()
    df = pd.read_csv(path)

    today_rows = [{"key": row["key"], "stand": row["stand"],
                   "significant": row["significant"], "factor": row["factor"]}
                  for _, row in df.iterrows()]
    log = _update_stability_log(stability_log_path, today_rows, today)

    factors: dict = {}
    n_real, n_neutral, n_pending = 0, 0, 0
    for _, row in df.iterrows():
        venue, stand = row["key"], row["stand"]
        factors.setdefault(venue, {})
        avg, streak = _qualifying_average(log, venue, stand, min_consecutive_days, max_average_window)
        if avg is not None:
            factors[venue][stand] = round(float(avg), 4)
            n_real += 1
        else:
            factors[venue][stand] = 0.0
            n_neutral += 1
            if 0 < streak < min_consecutive_days:
                n_pending += 1

    print(f"{label}: {n_real} real (stable {min_consecutive_days}+ day streak), "
          f"{n_neutral} neutral ({n_pending} of those mid-streak, not yet at {min_consecutive_days} days).")
    _print_almost_qualifying(log, min_consecutive_days)
    return factors


def export_weather_factors(path: str = "data/weather_factors_model.csv", today: str | None = None,
                            stability_log_path: str = WEATHER_STABILITY_LOG, label: str = "Weather factors",
                            min_consecutive_days: int = MIN_CONSECUTIVE_DAYS,
                            max_average_window: int = MAX_AVERAGE_WINDOW) -> dict:
    """Same reasoning as export_park_factors above — parameterized so a
    different prop's calibration can reuse this unchanged with its own
    separate stability history. Defaults match the original hardcoded
    behavior exactly."""
    today = today or date.today().isoformat()
    df = pd.read_csv(path)

    today_rows = [{"key": row["key"], "stand": row["stand"],
                   "significant": row["significant"], "factor": row["factor"]}
                  for _, row in df.iterrows()]
    log = _update_stability_log(stability_log_path, today_rows, today)

    factors: dict = {"L": {}, "R": {}}
    n_real, n_neutral, n_unparsed, n_pending = 0, 0, 0, 0
    for _, row in df.iterrows():
        m = WEATHER_KEY_RE.match(row["key"])
        if not m:
            n_unparsed += 1
            continue
        temp_bucket, wind_dir, speed_bucket = m.groups()
        stand = row["stand"]
        factors[stand].setdefault(temp_bucket, {}).setdefault(wind_dir, {})

        avg, streak = _qualifying_average(log, row["key"], stand, min_consecutive_days, max_average_window)
        if avg is not None:
            factors[stand][temp_bucket][wind_dir][speed_bucket] = round(float(avg), 4)
            n_real += 1
        else:
            factors[stand][temp_bucket][wind_dir][speed_bucket] = 0.0
            n_neutral += 1
            if 0 < streak < min_consecutive_days:
                n_pending += 1

    print(f"{label}: {n_real} real (stable {min_consecutive_days}+ day streak), "
          f"{n_neutral} neutral ({n_pending} mid-streak), {n_unparsed} keys failed to parse (should be 0).")
    _print_almost_qualifying(log, min_consecutive_days)
    return factors


if __name__ == "__main__":
    today = date.today().isoformat()

    park_factors = export_park_factors(today=today)
    with open("data/park_factors_live.json", "w") as f:
        json.dump(park_factors, f, indent=2)
    print(f"Saved {len(park_factors)} venues to data/park_factors_live.json")

    weather_factors = export_weather_factors(today=today)
    with open("data/weather_factors_live.json", "w") as f:
        json.dump(weather_factors, f, indent=2)
    total_buckets = sum(
        len(speeds) for hand in weather_factors.values()
        for winds in hand.values() for speeds in winds.values()
    )
    print(f"Saved {total_buckets} weather buckets to data/weather_factors_live.json")

    print("\nSample park factors (first 5):")
    for venue in list(park_factors)[:5]:
        print(f"  {venue}: {park_factors[venue]}")
