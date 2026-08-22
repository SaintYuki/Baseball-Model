"""
daily_update.py — one-command daily pipeline: pull any completed game-days
since the last cached pull, append them to the historical dataset, and
recalibrate the whole model (park factors, weather factors, batter power,
matchup) on the combined history.

WHAT'S NEW: also runs the Path 2 hits-prop calibration (calibrate_hits_model.py
+ export_hits_live_factors.py) and the total-bases calibration
(calibrate_tb_model.py + export_tb_live_factors.py) right after the HR one.
No separate data pull needed for either — all three calibrations read the
exact same cached batted-ball files, just targeting a different outcome.
Both newer props' steps are SOFT (non-fatal): if something in either
pipeline breaks, that should never block the actual daily HR picks, which
is the part of this that matters most. This also closes a real gap —
without it, a prop's stability-gating streak (each export_*_live_factors.py's
consecutive-day requirement) would only advance on days someone remembers to
run the scripts by hand, which defeats the point of a *consecutive*-day
requirement.

WHY THIS EXISTS: historical_data.py, calibrate_model.py, and
export_live_factors.py already compose cleanly on their own —
calibrate_model.py reloads EVERY data/batted_balls_*.parquet file on each
run and dedupes on exact pitch key (game_pk + at_bat_number + pitch_number),
so a new day's parquet file just needs to exist on disk for the next
recalibration to pick it up automatically, no merging required. This script
is the "figure out what's new, pull it, then run the other two in order"
wrapper, so the daily routine is one command instead of three, and instead
of guessing a date range by hand each morning.

WHAT COUNTS AS "NEW": every data/batted_balls_<start>_<end>.parquet file
already on disk gets parsed for its end date; the latest one found becomes
the day AFTER which we pull. Default end date is yesterday — Statcast data
for a day isn't reliably finalized until the following day, the same
reasoning historical_data.py's own pilot default already uses.

ALSO RUNS check_results.py FIRST: resolves yesterday's logged predictions
(from hr_scanner_auto.py's predictions_log.csv) against real outcomes
before doing anything else. This step is optional/non-fatal — if it fails
(e.g. no log yet on a fresh setup) the rest of the pipeline still runs.

RUN:
    python3 daily_update.py                        # pull since last cached day through yesterday, recalibrate
    python3 daily_update.py 2026-08-09 2026-08-09   # force a specific range, then recalibrate
    python3 daily_update.py --skip-pull             # recalibrate only, no new pull (e.g. after a code change)
"""

import glob
import os
import re
import subprocess
import sys
from datetime import date, timedelta

from historical_data import pull_batted_balls, pull_historical_weather, DATA_DIR

FILENAME_RE = re.compile(r"batted_balls_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")


def latest_cached_end_date():
    """Scans data/batted_balls_*.parquet filenames for the latest end date
    already pulled. Returns None if nothing's cached yet."""
    end_dates = []
    for f in glob.glob(os.path.join(DATA_DIR, "batted_balls_*.parquet")):
        m = FILENAME_RE.search(f)
        if m:
            end_dates.append(date.fromisoformat(m.group(2)))
    return max(end_dates) if end_dates else None


def pull_new_data(start_dt: str, end_dt: str):
    print(f"Pulling new batted-ball + weather data: {start_dt} to {end_dt}...")
    batted = pull_batted_balls(start_dt, end_dt)
    if batted.empty:
        print("  No batted balls returned for this range (off day, postponed games, or "
              "too-recent a date for Statcast to have finalized yet) — nothing to append.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"batted_balls_{start_dt}_{end_dt}.parquet")
    batted.to_parquet(out_path, index=False)
    print(f"  Saved {len(batted)} rows to {out_path}")

    unique_games = batted["game_pk"].dropna().unique().tolist()
    weather = pull_historical_weather(unique_games)

    matched = weather[weather["game_pk"].isin([int(g) for g in unique_games])]
    hit_rate = (matched["temp"].astype(str).str.strip() != "").mean() if not matched.empty else 0
    print(f"  Weather matched for {int(hit_rate * len(matched))}/{len(matched)} of the new games "
          f"({hit_rate*100:.0f}%).")
    if hit_rate < 0.8 and len(matched) > 0:
        print("  *** Weather hit rate is lower than usual for a completed day's games — worth a look "
              "before trusting today's recalibration. ***")


def run_step(cmd: list[str]):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"*** {cmd[0]} exited with code {result.returncode} — stopping the pipeline here, "
              f"not touching the live JSON files with a possibly-broken calibration. ***")
        sys.exit(result.returncode)


def run_step_soft(cmd: list[str]):
    """Same as run_step but doesn't halt the pipeline on failure — for steps
    that are useful but not load-bearing for the calibration itself (e.g.
    resolving yesterday's predictions shouldn't block today's recalibration
    just because, say, the log doesn't exist yet on a fresh setup)."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ({cmd[0]} exited with code {result.returncode} — continuing anyway, "
              f"this step is optional.)")


if __name__ == "__main__":
    run_step_soft([sys.executable, "check_results.py"])

    args = sys.argv[1:]
    skip_pull = "--skip-pull" in args
    args = [a for a in args if a != "--skip-pull"]

    if skip_pull:
        print("--skip-pull: recalibrating on whatever is already cached, no new data pull.")
    else:
        if len(args) == 2:
            start_dt, end_dt = args
        else:
            last_end = latest_cached_end_date()
            if last_end is None:
                print("No cached batted-ball files found in ./data/ — this script only APPENDS to an "
                      "existing history. Run historical_data.py once by hand first to establish a "
                      "starting dataset (e.g. python3 historical_data.py 2025-04-01 2025-09-30), "
                      "then use daily_update.py from here on.")
                sys.exit(1)

            yesterday = date.today() - timedelta(days=1)
            start = last_end + timedelta(days=1)
            if start > yesterday:
                print(f"Already caught up through {last_end.isoformat()} (yesterday was "
                      f"{yesterday.isoformat()}) — nothing new to pull. Recalibrating on the existing "
                      f"cache anyway, in case anything else (code, calibration logic) changed since "
                      f"the last run.")
                start_dt = end_dt = None
            else:
                start_dt, end_dt = start.isoformat(), yesterday.isoformat()

        if start_dt is not None:
            pull_new_data(start_dt, end_dt)

    run_step([sys.executable, "calibrate_model.py"])
    run_step([sys.executable, "export_live_factors.py"])

    print("\nHR calibration done — data/park_factors_live.json and data/weather_factors_live.json "
          "reflect the full history through this run.")

    # Path 2 pilot (hits prop) — soft/non-fatal on purpose. Reuses the exact same cached
    # batted-ball data just pulled/loaded above, no separate pull needed. If this newer
    # pipeline breaks for some reason, that should never block today's actual HR picks.
    run_step_soft([sys.executable, "calibrate_hits_model.py"])
    run_step_soft([sys.executable, "export_hits_live_factors.py"])

    # Path 2, prop 2 (total bases) — same soft/non-fatal reasoning, same shared data, no
    # separate pull. This one's calibration is heavier (game-level aggregation instead of
    # per-batted-ball) but the orchestration story is identical.
    run_step_soft([sys.executable, "calibrate_tb_model.py"])
    run_step_soft([sys.executable, "export_tb_live_factors.py"])

    print("\nDone.")
