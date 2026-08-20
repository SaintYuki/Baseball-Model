"""
export_hits_live_factors.py — the hits-prop version of export_live_factors.py.
Not a reimplementation: this is a thin wrapper around export_live_factors.py's
export_park_factors()/export_weather_factors(), which are already generic
enough (after adding stability_log_path/label parameters) to run this prop's
calibration output through the exact same significance + stability-gating
logic the HR pipeline uses — same rule, same reasoning, applied to a
completely separate stability history so the two props' significance logs
never mix.

WHY STABILITY-GATED FROM DAY ONE, UNLIKE HR'S FIRST VERSION: the HR pipeline
shipped ungated significance first and only added the consecutive-day
requirement later, after Tropicana Field's +79% one-day reading turned out
to have a CI spanning +12% to +181%. No reason to relearn that lesson here —
same MIN_CONSECUTIVE_DAYS / MAX_AVERAGE_WINDOW discipline applies from the
start, just tracked in its own log so a hits-specific streak can never be
padded or reset by HR's calibration history or vice versa.

TRANSITION COST, SAME AS BEFORE: the hits stability logs start empty, so
every factor exports as neutral for the first HITS_MIN_CONSECUTIVE_DAYS-1
runs, even the ones that looked robust in the very first calibration
(Fenway Park R, Petco Park L, etc.). One-time cost.

OUTPUT:
    data/hits_park_factors_live.json      {venue: {"L": factor, "R": factor}}
    data/hits_weather_factors_live.json   {"L": {temp: {wind_dir: {speed: factor}}}, "R": {...}}
    data/hits_park_stability_log.csv      running per-day significance history (hits, park)
    data/hits_weather_stability_log.csv   running per-day significance history (hits, weather)

RUN (after calibrate_hits_model.py has produced its CSVs):
    python3 export_hits_live_factors.py
"""

import json
from datetime import date

from export_live_factors import export_park_factors, export_weather_factors

# Separate from export_live_factors.py's HR constants, even though starting at the same
# values — same "avoid future coupling" reasoning as HIT_RATE_WINDOW_DAYS being its own
# constant in historical_features.py rather than reusing BATTER_POWER_WINDOW_DAYS.
HITS_MIN_CONSECUTIVE_DAYS = 3
HITS_MAX_AVERAGE_WINDOW = 14

HITS_PARK_STABILITY_LOG = "data/hits_park_stability_log.csv"
HITS_WEATHER_STABILITY_LOG = "data/hits_weather_stability_log.csv"


if __name__ == "__main__":
    today = date.today().isoformat()

    park_factors = export_park_factors(
        path="data/hits_park_factors_model.csv", today=today,
        stability_log_path=HITS_PARK_STABILITY_LOG, label="Hits park factors",
        min_consecutive_days=HITS_MIN_CONSECUTIVE_DAYS, max_average_window=HITS_MAX_AVERAGE_WINDOW,
    )
    with open("data/hits_park_factors_live.json", "w") as f:
        json.dump(park_factors, f, indent=2)
    print(f"Saved {len(park_factors)} venues to data/hits_park_factors_live.json")

    weather_factors = export_weather_factors(
        path="data/hits_weather_factors_model.csv", today=today,
        stability_log_path=HITS_WEATHER_STABILITY_LOG, label="Hits weather factors",
        min_consecutive_days=HITS_MIN_CONSECUTIVE_DAYS, max_average_window=HITS_MAX_AVERAGE_WINDOW,
    )
    with open("data/hits_weather_factors_live.json", "w") as f:
        json.dump(weather_factors, f, indent=2)
    total_buckets = sum(
        len(speeds) for hand in weather_factors.values()
        for winds in hand.values() for speeds in winds.values()
    )
    print(f"Saved {total_buckets} weather buckets to data/hits_weather_factors_live.json")

    print("\nSample hits park factors (first 5):")
    for venue in list(park_factors)[:5]:
        print(f"  {venue}: {park_factors[venue]}")
