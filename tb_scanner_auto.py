"""
tb_scanner_auto.py — Path 2, prop 2: the total-bases version of
hr_scanner_auto.py / hits_scanner_auto.py. Same two-stage pipeline (fast
trailing-rate scoring for everyone, then real pitch-level matchup data
layered on top), just scoring "2+ total bases tonight" using
data/tb_park_factors_live.json / data/tb_weather_factors_live.json.

WHAT'S REUSED, NOT DUPLICATED: load_todays_games, load_weather,
load_person_details, effective_bats, and apply_matchup_layer from
mlb_live_data.py — same genuinely prop-agnostic pieces every scanner_auto
script reuses.

WHAT'S DIFFERENT FROM THE HITS VERSION, ON PURPOSE:
  - No season-aggregate fallback, same reasoning as hits_scanner_auto.py:
    batters without enough trailing history get LEAGUE_AVERAGE_TB_RATE
    instead of a fabricated confident number.
  - tb_rate is NOT rescaled onto a 0-1 "eliteness" dial, same choice as
    hit_rate — real values already sit in an interpretable range.
  - No HTML report integration in THIS file — hr_scanner_auto.py is the
    one script that builds the combined page, and it imports the pieces it
    needs from here directly (same pattern it already uses for hits).
  - Runs its OWN schedule/roster/weather pull when run standalone,
    independent of the other scanner_auto scripts — same known,
    accepted inefficiency as hits_scanner_auto.py if run separately.

WORTH REMEMBERING FROM CALIBRATION, EVEN THOUGH IT DOESN'T CHANGE ANYTHING
HERE: the calibration's *outcome* (is_2plus_tb) was a whole-game
aggregate — that's already baked into the park/weather factors this script
reads. Scoring here is still per-batter-game like every other prop.

SETUP: pip install pybaseball MLB-StatsAPI pandas
RUN:
    python3 tb_scanner_auto.py               # today
    python3 tb_scanner_auto.py 2026-08-08     # a specific date
"""

import os
import sys
from datetime import date
from typing import Optional

import pandas as pd

from calibrate_weather import categorize_wind
from historical_features import load_batted_balls, add_batter_tb_rate, MIN_PRIOR_BATTED_BALLS_FOR_TB
from tb_scanner import (
    Batter, Game, rank_slate, rank_value_plays, build_parlays, print_report, is_known_venue,
)
from mlb_live_data import (
    load_todays_games, load_weather, load_person_details, effective_bats,
    apply_matchup_layer, TOP_K_FOR_MATCHUP,
)

LEAGUE_AVERAGE_TB_RATE = 0.51  # fallback for batters with no trailing tb-rate history yet.
                                # From the real theoretical estimate (league SLG / contact rate
                                # ≈ 0.40 / 0.78 ≈ 0.51) -- same "assume average, don't fabricate
                                # confidence" reasoning as LEAGUE_AVERAGE_HIT_RATE. Revisit once
                                # a real tb_scanner_auto.py run shows the actual trailing-rate
                                # distribution, same as every other placeholder in this project.


def load_tb_rates() -> dict:
    """trailing_by_id {mlbam_id: tb_rate} — point-in-time trailing total-bases
    rate per batter, from the same cached data/batted_balls_*.parquet history
    calibrate_tb_model.py already uses via add_batter_tb_rate. No live pull
    needed. Batters not in this dict (not enough trailing volume yet) get
    LEAGUE_AVERAGE_TB_RATE at the call site, not here."""
    import glob
    trailing_by_id: dict = {}
    if not glob.glob("data/batted_balls_*.parquet"):
        print("  (no cached batted-ball history found — run historical_data.py / daily_update.py "
              "first for trailing TB rates; everyone will use the league-average placeholder)")
        return trailing_by_id

    try:
        batted = load_batted_balls(None)
        with_rate = add_batter_tb_rate(batted)
        latest = (
            with_rate.dropna(subset=["batter_tb_rate"])
            .sort_values("game_date")
            .groupby("batter")
            .tail(1)
        )
        for _, row in latest.iterrows():
            trailing_by_id[int(row["batter"])] = round(float(row["batter_tb_rate"]), 3)

        if trailing_by_id:
            values = sorted(trailing_by_id.values())
            n = len(values)
            print(f"  Trailing TB rate: min={values[0]:.3f} median={values[n//2]:.3f} "
                  f"max={values[-1]:.3f} (real MLB TB-per-batted-ball is roughly 0.50-0.53 "
                  f"— sanity check against that)")
    except Exception as e:
        print(f"  (trailing TB-rate computation failed: {e} — everyone will use the "
              f"league-average placeholder)")

    return trailing_by_id


def load_team_batters_tb(team_id: int, team_abbr: str, trailing_by_id: dict,
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
        tb_rate = trailing_by_id.get(pid, LEAGUE_AVERAGE_TB_RATE)
        bat_side = details.get(pid, {}).get("bats", "R")
        batters.append(Batter(
            name=name, team=team_abbr, bats=effective_bats(bat_side, opp_throws),
            tb_rate=round(tb_rate, 3), mlbam_id=pid,
            opp_pitcher_id=opp_pitcher_id, opp_pitcher_name=opp_pitcher_name,
        ))
    return batters


TB_PREDICTIONS_LOG_PATH = "data/tb_predictions_log.csv"
TB_PREDICTIONS_LOG_COLUMNS = [
    "date", "mlbam_id", "game_pk", "player", "team", "game", "opp_pitcher",
    "park", "park_factor", "weather_adj", "matchup_adjustment",
    "tb_rate", "situational_boost", "score", "rank",
    "tb_2plus", "total_bases", "resolved",
]


def log_predictions_tb(ranked: list[dict], target_date: str):
    """Same pattern as log_predictions_hits, separate file — tb_2plus/
    total_bases/resolved start blank, to be filled in later by a
    TB-specific results resolver (not built yet)."""
    rows = [{
        "date": target_date, "mlbam_id": r["mlbam_id"], "game_pk": r["game_pk"],
        "player": r["player"], "team": r["team"], "game": r["game"], "opp_pitcher": r["opp_pitcher"],
        "park": r["park"], "park_factor": r["park_factor"], "weather_adj": r["weather_adj"],
        "matchup_adjustment": r["matchup_adjustment"], "tb_rate": r["tb_rate"],
        "situational_boost": r["situational_boost"], "score": r["score"], "rank": r["rank"],
        "tb_2plus": None, "total_bases": None, "resolved": False,
    } for r in ranked]
    new_df = pd.DataFrame(rows, columns=TB_PREDICTIONS_LOG_COLUMNS)

    os.makedirs("data", exist_ok=True)
    if os.path.exists(TB_PREDICTIONS_LOG_PATH):
        existing = pd.read_csv(TB_PREDICTIONS_LOG_PATH)
        existing = existing[existing["date"].astype(str) != str(target_date)]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(TB_PREDICTIONS_LOG_PATH, index=False)
    print(f"\nLogged {len(new_df)} TB predictions for {target_date} to {TB_PREDICTIONS_LOG_PATH} "
          f"({len(combined)} total rows across all dates). No results resolver for TB yet — "
          f"tb_2plus/total_bases stay blank until one exists.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    print(f"Pulling real MLB slate for {target} (TOTAL BASES)...\n")

    trailing_by_id = load_tb_rates()
    print(f"Loaded trailing TB rates for {len(trailing_by_id)} batters "
          f"(everyone else uses the {LEAGUE_AVERAGE_TB_RATE} league-average placeholder).\n")

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
        home_batters = load_team_batters_tb(g["home_id"], g["home_team"], trailing_by_id,
                                             g["away_pitcher_id"], g["away_pitcher_name"])
        away_batters = load_team_batters_tb(g["away_id"], g["away_team"], trailing_by_id,
                                             g["home_pitcher_id"], g["home_pitcher_name"])
        game.batters = home_batters + away_batters
        for b in game.batters:
            all_batters_by_id[b.mlbam_id] = b
        slate.append(game)

        venue_note = "" if is_known_venue(game.park) else \
            "  *** VENUE NOT IN tb_park_factors_live.json — check for a name mismatch/rename, " \
            "currently scoring as neutral ***"
        print(f"  {game.game_id()} @ {game.park} — {len(game.batters)} batters loaded, "
              f"wind={speed}mph dir={wind_dir}, temp={temp}, dome={is_dome}, "
              f"pitchers: {g['away_pitcher_name']} vs {g['home_pitcher_name']}{venue_note}")

    print()
    ranked = rank_slate(slate)

    apply_matchup_layer(ranked, all_batters_by_id, top_k=TOP_K_FOR_MATCHUP)

    ranked = rank_slate(slate)

    parlays = build_parlays(ranked, n_parlays=3, legs=3)

    value_plays = rank_value_plays(ranked, exclude_top_n=20, min_tb_rate=0.3, top_n=15)
    value_parlays = build_parlays(value_plays, n_parlays=2, legs=3)

    print_report(ranked, parlays, value_plays, value_parlays)

    log_predictions_tb(ranked, target)
