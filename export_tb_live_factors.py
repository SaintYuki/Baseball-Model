"""
export_tb_live_factors.py — the total-bases version of export_live_factors.py.
Not a reimplementation: this is a thin wrapper around export_live_factors.py's
export_park_factors()/export_weather_factors(), same as export_hits_live_factors.py —
running this prop's calibration output through the exact same significance +
stability-gating logic, applied to a completely separate stability history so
TB's significance log never mixes with HR's or hits'.

TRANSITION COST, SAME AS BEFORE: the TB stability logs start empty, so every
factor exports as neutral for the first TB_MIN_CONSECUTIVE_DAYS-1 runs, even
Coors Field — which just posted the single most robust park effect this whole
project has calibrated (+36%, CI [+21%, +50%], real and well-known: altitude
carry is the most famous park effect in baseball). It'll still take 3
consecutive significant calibrations before it goes live, same rule as
everything else. One-time cost, not a reason to bend the rule for one park.

OUTPUT:
    data/tb_park_factors_live.json      {venue: {"L": factor, "R": factor}}
    data/tb_weather_factors_live.json   {"L": {temp: {wind_dir: {speed: factor}}}, "R": {...}}
    data/tb_park_stability_log.csv      running per-day significance history (TB, park)
    data/tb_weather_stability_log.csv   running per-day significance history (TB, weather)

RUN (after calibrate_tb_model.py has produced its CSVs):
    python3 export_tb_live_factors.py
"""

import json
from datetime import date

from export_live_factors import export_park_factors, export_weather_factors

# Separate from HR's and hits' constants, even though starting at the same values —
# same "avoid future coupling" reasoning used for every prop-specific constant so far.
TB_MIN_CONSECUTIVE_DAYS = 3
TB_MAX_AVERAGE_WINDOW = 14

TB_PARK_STABILITY_LOG = "data/tb_park_stability_log.csv"
TB_WEATHER_STABILITY_LOG = "data/tb_weather_stability_log.csv"


if __name__ == "__main__":
    today = date.today().isoformat()

    park_factors = export_park_factors(
        path="data/tb_park_factors_model.csv", today=today,
        stability_log_path=TB_PARK_STABILITY_LOG, label="TB park factors",
        min_consecutive_days=TB_MIN_CONSECUTIVE_DAYS, max_average_window=TB_MAX_AVERAGE_WINDOW,
    )
    with open("data/tb_park_factors_live.json", "w") as f:
        json.dump(park_factors, f, indent=2)
    print(f"Saved {len(park_factors)} venues to data/tb_park_factors_live.json")

    weather_factors = export_weather_factors(
        path="data/tb_weather_factors_model.csv", today=today,
        stability_log_path=TB_WEATHER_STABILITY_LOG, label="TB weather factors",
        min_consecutive_days=TB_MIN_CONSECUTIVE_DAYS, max_average_window=TB_MAX_AVERAGE_WINDOW,
    )
    with open("data/tb_weather_factors_live.json", "w") as f:
        json.dump(weather_factors, f, indent=2)
    total_buckets = sum(
        len(speeds) for hand in weather_factors.values()
        for winds in hand.values() for speeds in winds.values()
    )
    print(f"Saved {total_buckets} weather buckets to data/tb_weather_factors_live.json")

    print("\nSample TB park factors (first 5):")
    for venue in list(park_factors)[:5]:
        print(f"  {venue}: {park_factors[venue]}")
