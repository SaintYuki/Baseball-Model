"""
hits_scanner_auto.py — Path 2 pilot: the hits-prop version of
hr_scanner_auto.py. Same two-stage pipeline (fast trailing-rate scoring for
everyone, then real pitch-level matchup data layered on top), just scoring
"1+ hit tonight" using data/hits_park_factors_live.json /
data/hits_weather_factors_live.json instead of the HR files.

WHAT'S REUSED FROM hr_scanner_auto.py, NOT DUPLICATED: load_todays_games,
load_weather, load_person_details, effective_bats, and apply_matchup_layer
are all genuinely prop-agnostic — schedule/weather/handedness pulls don't
care what prop you're scoring, and apply_matchup_layer only ever touches
b.matchup_adjustment and b.opp_pitcher_id, never power_score or hit_rate
specifically. Imported directly rather than copy-pasted.

WHAT'S DIFFERENT FROM THE HR VERSION, ON PURPOSE:
  - No season-aggregate fallback (HR's load_power_scores pulls a full Savant
    season leaderboard as a backup for thin-history batters). Skipped here
    deliberately rather than built untested — hits happen ~10x more often
    than HRs per batted ball, so MIN_PRIOR_BATTED_BALLS_FOR_HITS is cleared
    much faster in real terms; the population actually needing a fallback
    is smaller. Batters without enough trailing history get
    LEAGUE_AVERAGE_HIT_RATE instead — an honest "we don't have a real read,
    assume average" placeholder, not a fabricated confident number.
  - hit_rate is NOT rescaled onto a 0-1 "eliteness" dial the way power_score
    is (via TRAILING_HR_RATE_ELITE). Real hit rates already sit in a
    naturally interpretable ~0.15-0.45 range, unlike HR rates (~0.00-0.15),
    so compressing them further isn't obviously worth the extra layer of
    indirection. Uses the raw trailing rate directly.
  - No HTML report integration yet (generate_report_html.py is HR-specific).
    Terminal report + a separate predictions log
    (data/hits_predictions_log.csv) only, for now.
  - Runs its OWN schedule/roster/weather pull, independent of
    hr_scanner_auto.py's — meaning if both run the same day, the same live
    API calls happen twice. Real, known inefficiency, not a correctness
    issue; a future refactor could share one pull across both props. Not
    worth the coupling risk to fix before this has even run once for real.

SETUP: pip install pybaseball MLB-StatsAPI pandas
RUN:
    python3 hits_scanner_auto.py               # today
    python3 hits_scanner_auto.py 2026-08-08     # a specific date
"""

import os
import sys
from datetime import date
from typing import Optional

import pandas as pd

from calibrate_weather import categorize_wind
from historical_features import load_batted_balls, add_batter_hit_rate, MIN_PRIOR_BATTED_BALLS_FOR_HITS
from hits_scanner import (
    Batter, Game, rank_slate, rank_value_plays, build_parlays, print_report, is_known_venue,
)
from mlb_live_data import (
    load_todays_games, load_weather, load_person_details, effective_bats,
    apply_matchup_layer, TOP_K_FOR_MATCHUP,
)

LEAGUE_AVERAGE_HIT_RATE = 0.30  # fallback for batters with no trailing hit-rate history yet.
                                 # Matches both the theoretical estimate (league AVG / contact
                                 # rate ≈ 0.314) and the real observed trailing-rate median from
                                 # actual season data (~0.326-0.327, seen consistently) closely
                                 # enough to trust as a genuine "assume average" placeholder.


def load_hit_rates() -> dict:
    """trailing_by_id {mlbam_id: hit_rate} — point-in-time trailing hit rate
    per batter, from the same cached data/batted_balls_*.parquet history
    calibrate_hits_model.py already uses via add_batter_hit_rate. No live
    pull needed. Batters not in this dict (not enough trailing volume yet)
    get LEAGUE_AVERAGE_HIT_RATE at the call site, not here."""
    import glob
    trailing_by_id: dict = {}
    if not glob.glob("data/batted_balls_*.parquet"):
        print("  (no cached batted-ball history found — run historical_data.py / daily_update.py "
              "first for trailing hit rates; everyone will use the league-average placeholder)")
        return trailing_by_id

    try:
        batted = load_batted_balls(None)
        with_rate = add_batter_hit_rate(batted)
        latest = (
            with_rate.dropna(subset=["batter_hit_rate"])
            .sort_values("game_date")
            .groupby("batter")
            .tail(1)
        )
        for _, row in latest.iterrows():
            trailing_by_id[int(row["batter"])] = round(float(row["batter_hit_rate"]), 3)

        if trailing_by_id:
            values = sorted(trailing_by_id.values())
            n = len(values)
            print(f"  Trailing hit rate: min={values[0]:.3f} median={values[n//2]:.3f} "
                  f"max={values[-1]:.3f} (real MLB hits-per-batted-ball is roughly 0.31-0.33 "
                  f"— sanity check against that)")
    except Exception as e:
        print(f"  (trailing hit-rate computation failed: {e} — everyone will use the "
              f"league-average placeholder)")

    return trailing_by_id


def load_team_batters_hits(team_id: int, team_abbr: str, trailing_by_id: dict,
                            opp_pitcher_id: Optional[int], opp_pitcher_name: str) -> list[Batter]:
    import statsapi
    roster = statsapi.get("team_roster", {"teamId": team_id, "rosterType": "active"})
    non_pitchers = [p for p in roster.get("roster", []) if p.get("position", {}).get("abbreviation") != "P"]

    ids_to_fetch = [p["person"]["id"] for p in non_pitchers]
    if opp_pitcher_id:
        ids_to_fetch.append(opp_pitcher_id)
    details = load_person_details(ids_to_fetch)
    opp_throws = details.get(opp_pitcher_id, {}).get("throws", "") if opp_pitcher_id else ""

    batters = []
    for p in non_pitchers:
        pid = p["person"]["id"]
        name = p["person"]["fullName"]
        hit_rate = trailing_by_id.get(pid, LEAGUE_AVERAGE_HIT_RATE)
        bat_side = details.get(pid, {}).get("bats", "R")
        batters.append(Batter(
            name=name, team=team_abbr, bats=effective_bats(bat_side, opp_throws),
            hit_rate=round(hit_rate, 3), mlbam_id=pid,
            opp_pitcher_id=opp_pitcher_id, opp_pitcher_name=opp_pitcher_name,
        ))
    return batters


HITS_PREDICTIONS_LOG_PATH = "data/hits_predictions_log.csv"
HITS_PREDICTIONS_LOG_COLUMNS = [
    "date", "mlbam_id", "game_pk", "player", "team", "game", "opp_pitcher",
    "park", "park_factor", "weather_adj", "matchup_adjustment",
    "hit_rate", "situational_boost", "score", "rank",
    "hit_1plus", "hit_count", "resolved",
]


def log_predictions_hits(ranked: list[dict], target_date: str):
    """Same pattern as hr_scanner_auto.log_predictions, separate file —
    hit_1plus/hit_count/resolved start blank, to be filled in later by a
    hits-specific results resolver (not built yet)."""
    rows = [{
        "date": target_date, "mlbam_id": r["mlbam_id"], "game_pk": r["game_pk"],
        "player": r["player"], "team": r["team"], "game": r["game"], "opp_pitcher": r["opp_pitcher"],
        "park": r["park"], "park_factor": r["park_factor"], "weather_adj": r["weather_adj"],
        "matchup_adjustment": r["matchup_adjustment"], "hit_rate": r["hit_rate"],
        "situational_boost": r["situational_boost"], "score": r["score"], "rank": r["rank"],
        "hit_1plus": None, "hit_count": None, "resolved": False,
    } for r in ranked]
    new_df = pd.DataFrame(rows, columns=HITS_PREDICTIONS_LOG_COLUMNS)

    os.makedirs("data", exist_ok=True)
    if os.path.exists(HITS_PREDICTIONS_LOG_PATH):
        existing = pd.read_csv(HITS_PREDICTIONS_LOG_PATH)
        existing = existing[existing["date"].astype(str) != str(target_date)]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(HITS_PREDICTIONS_LOG_PATH, index=False)
    print(f"\nLogged {len(new_df)} hits predictions for {target_date} to {HITS_PREDICTIONS_LOG_PATH} "
          f"({len(combined)} total rows across all dates). No results resolver for hits yet — "
          f"hit_1plus/hit_count stay blank until one exists.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    print(f"Pulling real MLB slate for {target} (HITS)...\n")

    trailing_by_id = load_hit_rates()
    print(f"Loaded trailing hit rates for {len(trailing_by_id)} batters "
          f"(everyone else uses the {LEAGUE_AVERAGE_HIT_RATE} league-average placeholder).\n")

    todays_games = load_todays_games(target)
    print(f"Found {len(todays_games)} games.\n")

    slate = []
    all_batters_by_id = {}
    for g in todays_games:
        wx = load_weather(g["game_pk"])
        speed, wind_dir = categorize_wind(wx.get("wind", ""))
        temp = wx.get("temp")
        is_dome = wx.get("condition", "").lower() in ("roof closed", "dome")

        game = Game(
            home_team=g["home_team"], away_team=g["away_team"], park=g["venue"],
            is_dome=is_dome, wind_speed_mph=speed, wind_dir=wind_dir, temp=temp,
            home_pitcher_id=g["home_pitcher_id"], home_pitcher_name=g["home_pitcher_name"],
            away_pitcher_id=g["away_pitcher_id"], away_pitcher_name=g["away_pitcher_name"],
            game_pk=g["game_pk"],
        )
        home_batters = load_team_batters_hits(g["home_id"], g["home_team"], trailing_by_id,
                                               g["away_pitcher_id"], g["away_pitcher_name"])
        away_batters = load_team_batters_hits(g["away_id"], g["away_team"], trailing_by_id,
                                               g["home_pitcher_id"], g["home_pitcher_name"])
        game.batters = home_batters + away_batters
        for b in game.batters:
            all_batters_by_id[b.mlbam_id] = b
        slate.append(game)

        venue_note = "" if is_known_venue(game.park) else \
            "  *** VENUE NOT IN hits_park_factors_live.json — check for a name mismatch/rename, " \
            "currently scoring as neutral ***"
        print(f"  {game.game_id()} @ {game.park} — {len(game.batters)} batters loaded, "
              f"wind={speed}mph dir={wind_dir}, temp={temp}, dome={is_dome}, "
              f"pitchers: {g['away_pitcher_name']} vs {g['home_pitcher_name']}{venue_note}")

    print()
    ranked = rank_slate(slate)

    apply_matchup_layer(ranked, all_batters_by_id, top_k=TOP_K_FOR_MATCHUP)

    ranked = rank_slate(slate)

    parlays = build_parlays(ranked, n_parlays=3, legs=3)

    value_plays = rank_value_plays(ranked, exclude_top_n=20, min_hit_rate=0.25, top_n=15)
    value_parlays = build_parlays(value_plays, n_parlays=2, legs=3)

    print_report(ranked, parlays, value_plays, value_parlays)

    log_predictions_hits(ranked, target)
