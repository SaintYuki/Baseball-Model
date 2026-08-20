"""
hr_scanner_auto.py — automated data pull for hr_scanner.py (v6)

WHAT'S NEW IN V6
  - Stage 2 (pitch-level matchup) no longer caps at the top 60 batters by
    default — it now pulls for the whole slate. The cap was a pure
    computational shortcut, not a claim that matchup edges only exist among
    already-high-scoring batters; in practice it meant the Value Board could
    only ever surface someone already in the top 60 by raw score, so a real
    matchup edge sitting at rank 90 was invisible by construction. Pitcher
    profiles are still cached per starter (bounded by ~15-30 starters/day),
    so this only scales the batter-side pull, roughly 60 -> ~390 for a full
    slate — not a multiplicative blowup. TOP_K_FOR_MATCHUP is still a real
    parameter (set it back to an int to re-cap) if this turns out to be too
    slow or trips a Savant rate limit on a real run.

WHAT'S NEW IN V5
  - power_score is now trailing 21-day HR rate (real recent form), not a
    season-cumulative barrel rate. The old season-aggregate version barely
    moved day to day once a player had a few hundred batted balls this
    season (K=100 shrinkage means one more game's ~4 batted balls changes
    almost nothing) — that's why the same handful of names kept sitting at
    the top of the board regardless of the actual slate. It's also the same
    metric and window (BATTER_POWER_WINDOW_DAYS, MIN_PRIOR_BATTED_BALLS)
    that calibrate_model.py already uses via historical_features.py, so live
    scoring now matches what the park/weather/matchup effects were actually
    calibrated against, instead of using a different metric entirely.
    Computed from the local cached parquet files daily_update.py already
    maintains — no new live pull needed. Falls back to the old season-
    aggregate barrel-rate approach for anyone without enough recent batted-
    ball volume (rookies, September call-ups, IL returners), matched by
    mlbam_id instead of name to avoid the accent/suffix name-matching
    fragility the old fallback-only version had.

WHAT'S NEW IN V4
  - Every run now logs its full ranked slate to data/predictions_log.csv
    (see log_predictions()). This is what lets check_results.py later verify
    which predictions actually came true, build a real score-to-HR-rate
    calibration, and check whether the Value Board picks are worth anything.
    Re-running for a date that's already logged replaces that date's rows
    rather than duplicating them.
  - Game now carries game_pk, so check_results.py can tell a postponed game
    apart from a game that happened with no home runs.

WHAT'S NEW IN V3
  - Wind is now categorized with calibrate_weather.categorize_wind() instead
    of the old parse_wind() in/out/None heuristic. That's the exact same
    parser calibrate_weather.py used to build data/weather_factors_live.json,
    so a live "7 mph, Out To LF" reading buckets into the identical "Out"
    category the calibration measured — parse_wind's cruder in/out/None
    split would have silently mismatched Calm/Cross/Varies games against
    buckets that don't exist under those labels.
  - temp now gets pulled through to the Game object too, since weather_adj
    is keyed on (handedness, temp bucket, wind dir, wind speed bucket).
  - Each game line now flags if its venue isn't in the calibrated park
    factors file at all (as opposed to being in it with a 0.0) — catches a
    park-rename mismatch instead of silently scoring it as "no park effect."

WHAT'S NEW IN V2
  - Pulls real probable-pitcher MLBAM IDs (not just names) straight from the
    MLB Stats API's raw schedule endpoint — statsapi's wrapper only exposes
    pitcher names, so this calls statsapi.get() directly instead.
  - Two-stage pipeline:
      Stage 1 (fast, everyone): trailing power score x park factor x
        weather. Cheap — no per-player calls beyond the local data already
        cached.
      Stage 2 (slow, top 60 only): pulls real recent pitch-by-pitch data for
        each of the top 60 batters AND their opposing starter, and layers in
        the matchup_adjustment from pitch_matchup.py. Everyone outside the
        top 60 keeps their Stage 1 score untouched (matchup = neutral).
    Stage 2 is gated to the top 60 on purpose — pulling per-batter Statcast
    data for all ~390 batters on a 15-game slate would mean hundreds of
    requests hammering Savant and take a long time. This keeps it to a
    couple hundred calls at most, with pitcher profiles cached and reused
    across every batter facing that same pitcher, plus a short delay between
    requests to be polite to Savant's servers.

SETUP: pip install pybaseball MLB-StatsAPI pandas
RUN:
    python3 hr_scanner_auto.py               # today
    python3 hr_scanner_auto.py 2026-08-08     # a specific date
"""

import glob
import os
import sys
import time
from datetime import date
from typing import Optional

import pandas as pd
import statsapi
import pybaseball as pyb

from calibrate_weather import categorize_wind
from generate_report_html import render_html
from historical_features import load_batted_balls, add_batter_power
from hr_scanner import (
    Batter, Game, rank_slate, rank_value_plays, build_parlays, print_report, is_known_venue,
)
from pitch_matchup import (
    get_batter_pitch_profile, get_pitcher_pitch_profile, matchup_score,
)

TOP_K_FOR_MATCHUP = None    # None = pull matchup data for every batter on the slate (not just
                             # the top N by base score). Was capped at 60 originally to keep the
                             # per-player Savant pulls manageable, but that cap meant the Value
                             # Board could only ever surface someone already in the top 60 by raw
                             # score — a real matchup edge sitting at rank 90 was structurally
                             # invisible. Pitcher-side calls are cached per starter regardless
                             # (bounded by ~15-30 starters/day, not by how many batters we score),
                             # so uncapping this only scales the batter-side pull, roughly 60->~390
                             # for a full slate. Set back to an int (e.g. 60) if this turns out to
                             # be too slow or trips a Savant rate limit on a real run.
REQUEST_DELAY_SEC = 0.4     # be polite to Savant between per-player pulls
TRAILING_HR_RATE_ELITE = 0.13  # trailing HR-rate-per-batted-ball treated as "maxed out" (~1.0)
                                # on the power_score dial. Raised from an initial 0.09 after
                                # multiple unrelated players independently hit that cap in the
                                # same run — that's a sign the bar was too easy to clear, not
                                # that they're all equally "maxed hot." Still a guess pending a
                                # real look at the raw trailing-rate distribution (now printed
                                # by load_power_scores) — tune again once you've seen real numbers.


def load_power_scores(year: int) -> tuple[dict, dict]:
    """Two power-score sources, layered:

    trailing_by_id {mlbam_id: power_score} — real recent form: trailing
    21-day HR rate per batted ball, computed from the local cached
    data/batted_balls_*.parquet history (same computation, same window,
    calibrate_model.py already runs via historical_features.add_batter_power).
    Keyed by mlbam_id, which matches the roster API's player id directly —
    no name-matching involved. Empty if no cached history exists yet.

    season_by_name {lowercased full name: power_score} — the original
    season-cumulative barrel-rate approach, unchanged, used only as a
    fallback for batters without enough recent batted-ball volume to trust
    a trailing rate (rookies, call-ups, IL returners). A call-up with 12
    batted balls and a hot streak won't rank as 'elite' off pure noise, but
    doesn't get excluded outright either — late in the season, rookies and
    call-ups ARE the slate some nights.

    load_team_batters checks trailing_by_id first; only falls through to
    season_by_name if the batter isn't in the trailing set at all."""
    df = pyb.statcast_batter_exitvelo_barrels(year, minBBE=1)
    name_col = "last_name, first_name"

    league_avg = (df["brl_percent"] / 100 * df["attempts"]).sum() / df["attempts"].sum()
    K = 100  # bigger K = more conservative on small samples

    season_by_name = {}
    for _, row in df.iterrows():
        raw_name = str(row.get(name_col, "")).strip()
        if not raw_name or "," not in raw_name:
            continue
        last, first = raw_name.split(",", 1)
        name = f"{first.strip()} {last.strip()}".lower()

        n = row.get("attempts", 0) or 0
        raw_rate = (row.get("brl_percent", 0) or 0) / 100
        shrunk_rate = (n * raw_rate + K * league_avg) / (n + K)
        season_by_name[name] = min(shrunk_rate / 0.25, 1.0)

    trailing_by_id: dict = {}
    if not glob.glob("data/batted_balls_*.parquet"):
        print("  (no cached batted-ball history found — run historical_data.py / daily_update.py "
              "first for trailing-form power scores; using season-aggregate only for now)")
    else:
        try:
            batted = load_batted_balls(None)
            with_power = add_batter_power(batted)
            latest = (
                with_power.dropna(subset=["batter_power"])
                .sort_values("game_date")
                .groupby("batter")
                .tail(1)
            )
            for _, row in latest.iterrows():
                trailing_by_id[int(row["batter"])] = min(row["batter_power"] / TRAILING_HR_RATE_ELITE, 1.0)

            # DIAGNOSTIC — this is a brand-new metric, worth eyeballing the real distribution
            # rather than trusting it blind. In particular: a cluster of exact 0.0 values among
            # players who should clearly have some recent power is a sign something's wrong
            # upstream (a data gap, a merge issue) rather than a real universal cold streak.
            if trailing_by_id:
                values = sorted(trailing_by_id.values())
                n = len(values)
                n_zero = sum(1 for v in values if v == 0.0)
                n_capped = sum(1 for v in values if v >= 1.0)
                raw = latest["batter_power"]
                print(f"  Trailing power (rescaled 0-1): min={values[0]:.3f} "
                      f"median={values[n//2]:.3f} max={values[-1]:.3f}")
                print(f"  {n_zero}/{n} batters show EXACTLY 0.0 trailing power — if this is a "
                      f"large chunk and includes recognizable everyday players, that's a sign of "
                      f"a real data gap, not a genuine simultaneous HR drought.")
                print(f"  {n_capped}/{n} batters are capped at 1.0 (raise TRAILING_HR_RATE_ELITE "
                      f"if this is more than a handful — losing separation between 'hot' and "
                      f"'extremely hot' isn't useful).")
                print(f"  Raw trailing HR rate (pre-rescale): min={raw.min():.4f} "
                      f"median={raw.median():.4f} max={raw.max():.4f} "
                      f"(MLB full-season average is roughly 0.03 — sanity check against that)")
        except Exception as e:
            print(f"  (trailing power computation failed: {e} — falling back to season-aggregate only)")

    return trailing_by_id, season_by_name


def load_todays_games(target_date: str) -> list[dict]:
    """Raw schedule call instead of statsapi.schedule() — the wrapper drops
    probable-pitcher IDs and only keeps names, and we need the IDs for the
    matchup pull."""
    hydrate = "probablePitcher,linescore"
    r = statsapi.get("schedule", {"sportId": 1, "date": target_date, "hydrate": hydrate})
    games = []
    for d in r.get("dates", []):
        for g in d.get("games", []):
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            home_p = home.get("probablePitcher", {}) or {}
            away_p = away.get("probablePitcher", {}) or {}
            games.append({
                "game_pk": g["gamePk"],
                "home_team": home["team"].get("name", "???"),
                "away_team": away["team"].get("name", "???"),
                "home_id": home["team"]["id"],
                "away_id": away["team"]["id"],
                "venue": g.get("venue", {}).get("name", ""),
                "home_pitcher_id": home_p.get("id"),
                "home_pitcher_name": home_p.get("fullName", ""),
                "away_pitcher_id": away_p.get("id"),
                "away_pitcher_name": away_p.get("fullName", ""),
            })
    return games


def load_weather(game_pk: int) -> dict:
    try:
        feed = statsapi.get("game", {"gamePk": game_pk})
        wx = feed.get("gameData", {}).get("weather", {})
        return {"condition": wx.get("condition", ""), "temp": wx.get("temp"), "wind": wx.get("wind", "")}
    except Exception as e:
        print(f"  (weather unavailable for game {game_pk}: {e})")
        return {}


def load_person_details(person_ids: list[int]) -> dict:
    """Batch-fetches real bat side + throwing hand for a list of MLBAM person
    ids in ONE call, instead of one request per player. Uses the 'people'
    endpoint (plural) — NOT 'person' (singular), which requires a single
    personId as a path parameter and can't batch. 'people' takes personIds
    as a comma-separated query parameter instead. Returns
    {id: {'bats': 'L'/'R'/'S', 'throws': 'L'/'R'}}. Falls back to an empty
    dict on failure so callers degrade to the 'R' default rather than crash.

    SANITY CHECK THIS ONCE LIVE: I can't hit the Stats API from my sandbox,
    so the first real run is the first time this exact batched-personIds
    call has actually been exercised — glance at a couple of known lefties
    (Ohtani, Freeman) in the printed slate and confirm they come back 'L'."""
    if not person_ids:
        return {}
    ids_param = ",".join(str(i) for i in sorted(set(person_ids)))
    try:
        r = statsapi.get("people", {"personIds": ids_param})
    except Exception as e:
        print(f"  (batch handedness pull failed: {e} — defaulting affected players to bats='R')")
        return {}
    details = {}
    for person in r.get("people", []):
        details[person["id"]] = {
            "bats": person.get("batSide", {}).get("code", "R"),
            "throws": person.get("pitchHand", {}).get("code", "R"),
        }
    return details


def effective_bats(bat_side: str, opp_throws: str) -> str:
    """Resolves a batter's effective handedness for scoring purposes.
    Switch hitters ('S') take the standard platoon side: left against a
    right-handed pitcher, right against a left-handed pitcher. Non-switch
    hitters just use their real side. Unknown pitcher hand falls back to
    'R' — same default the old hardcoded version used, but now only for
    the genuinely-unknown case instead of for every single batter."""
    if bat_side != "S":
        return "L" if bat_side == "L" else "R"
    if opp_throws == "L":
        return "R"
    if opp_throws == "R":
        return "L"
    return "R"


def load_team_batters(team_id: int, team_abbr: str, trailing_by_id: dict, season_by_name: dict,
                       opp_pitcher_id: Optional[int], opp_pitcher_name: str) -> list[Batter]:
    roster = statsapi.get("team_roster", {"teamId": team_id, "rosterType": "active"})
    non_pitchers = [p for p in roster.get("roster", []) if p.get("position", {}).get("abbreviation") != "P"]

    ids_to_fetch = [p["person"]["id"] for p in non_pitchers]
    if opp_pitcher_id:
        ids_to_fetch.append(opp_pitcher_id)  # need the starter's throwing hand to resolve switch hitters
    details = load_person_details(ids_to_fetch)
    opp_throws = details.get(opp_pitcher_id, {}).get("throws", "") if opp_pitcher_id else ""

    batters = []
    for p in non_pitchers:
        pid = p["person"]["id"]
        name = p["person"]["fullName"]
        if pid in trailing_by_id:
            power = trailing_by_id[pid]
        else:
            power = season_by_name.get(name.lower(), 0.30)
        bat_side = details.get(pid, {}).get("bats", "R")
        batters.append(Batter(
            name=name, team=team_abbr, bats=effective_bats(bat_side, opp_throws),
            power_score=round(power, 3), mlbam_id=pid,
            opp_pitcher_id=opp_pitcher_id, opp_pitcher_name=opp_pitcher_name,
        ))
    return batters


def apply_matchup_layer(ranked: list[dict], all_batters_by_id: dict,
                         top_k: Optional[int] = TOP_K_FOR_MATCHUP):
    """Mutates matchup_adjustment on the top_k Batter objects in place (or
    on everyone, if top_k is None), using real pitch-by-pitch data. Pitcher
    profiles are cached so a starter facing 13 opposing batters only gets
    pulled once, not 13 times — this cache is what keeps an uncapped run
    from scaling anywhere near as badly as "one call per batter" sounds;
    it's really "one call per batter, plus ~15-30 calls total for the
    day's starters," not multiplicatively worse per batter added.

    Looks batters up by mlbam_id, not name — name collisions are real (MLB
    currently has two active players named Max Muncy, on different teams).
    A name-keyed dict would silently merge them into one entry, so the
    later-loaded one is the only one reachable, its pull runs twice for no
    reason, and the other player's real matchup data never gets applied."""
    pitcher_cache: dict[int, object] = {}
    top_rows = ranked if top_k is None else ranked[:top_k]

    scope = f"all {len(top_rows)} batters" if top_k is None else f"the top {len(top_rows)} batters"
    print(f"\nPulling pitch-level matchup data for {scope} "
          f"(this is the slow part — one Savant call per batter, cached per pitcher)...")

    for i, row in enumerate(top_rows, 1):
        b = all_batters_by_id.get(row["mlbam_id"])
        if b is None or b.mlbam_id is None or b.opp_pitcher_id is None:
            continue

        if b.opp_pitcher_id not in pitcher_cache:
            try:
                pitcher_cache[b.opp_pitcher_id] = get_pitcher_pitch_profile(b.opp_pitcher_id)
            except Exception as e:
                print(f"  ({b.opp_pitcher_name} pitcher pull failed: {e})")
                pitcher_cache[b.opp_pitcher_id] = None
            time.sleep(REQUEST_DELAY_SEC)
        pitcher_profile = pitcher_cache[b.opp_pitcher_id]
        if pitcher_profile is None or pitcher_profile.empty:
            continue

        try:
            batter_profile = get_batter_pitch_profile(b.mlbam_id)
        except Exception as e:
            print(f"  ({b.name} batter pull failed: {e})")
            continue
        time.sleep(REQUEST_DELAY_SEC)

        result = matchup_score(batter_profile, pitcher_profile)
        b.matchup_adjustment = result["adjustment"]

        print(f"  [{i}/{len(top_rows)}] {b.name:<22} vs {b.opp_pitcher_name:<20} "
              f"matchup {result['adjustment']:+.3f}")


PREDICTIONS_LOG_PATH = "data/predictions_log.csv"
PREDICTIONS_LOG_COLUMNS = [
    "date", "mlbam_id", "game_pk", "player", "team", "game", "opp_pitcher",
    "park", "park_factor", "weather_adj", "matchup_adjustment",
    "power_score", "situational_boost", "score", "rank",
    "hit_hr", "hr_count", "resolved",
]


def log_predictions(ranked: list[dict], target_date: str):
    """Appends today's full ranked slate to data/predictions_log.csv, so
    check_results.py can later verify what actually happened. hit_hr /
    hr_count / resolved start blank — check_results.py fills them in once
    the date is in the past and Statcast has finalized it.

    Re-running hr_scanner_auto.py for a date that's already in the log
    replaces that date's rows instead of duplicating them, so the log
    always reflects the most recent run for a given day."""
    rows = [{
        "date": target_date,
        "mlbam_id": r["mlbam_id"],
        "game_pk": r["game_pk"],
        "player": r["player"],
        "team": r["team"],
        "game": r["game"],
        "opp_pitcher": r["opp_pitcher"],
        "park": r["park"],
        "park_factor": r["park_factor"],
        "weather_adj": r["weather_adj"],
        "matchup_adjustment": r["matchup_adjustment"],
        "power_score": r["power_score"],
        "situational_boost": r["situational_boost"],
        "score": r["score"],
        "rank": r["rank"],
        "hit_hr": None,
        "hr_count": None,
        "resolved": False,
    } for r in ranked]
    new_df = pd.DataFrame(rows, columns=PREDICTIONS_LOG_COLUMNS)

    os.makedirs("data", exist_ok=True)
    if os.path.exists(PREDICTIONS_LOG_PATH):
        existing = pd.read_csv(PREDICTIONS_LOG_PATH)
        existing = existing[existing["date"].astype(str) != str(target_date)]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(PREDICTIONS_LOG_PATH, index=False)
    print(f"\nLogged {len(new_df)} predictions for {target_date} to {PREDICTIONS_LOG_PATH} "
          f"({len(combined)} total rows across all dates). Run check_results.py once these "
          f"games are final to see how the model actually did.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    print(f"Pulling real MLB slate for {target}...\n")

    trailing_by_id, season_by_name = load_power_scores(date.today().year)
    print(f"Loaded trailing (recent-form) power scores for {len(trailing_by_id)} batters, "
          f"season-aggregate fallback for {len(season_by_name)} batters.\n")

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
        # home batters face the AWAY pitcher, away batters face the HOME pitcher
        home_batters = load_team_batters(g["home_id"], g["home_team"], trailing_by_id, season_by_name,
                                          g["away_pitcher_id"], g["away_pitcher_name"])
        away_batters = load_team_batters(g["away_id"], g["away_team"], trailing_by_id, season_by_name,
                                          g["home_pitcher_id"], g["home_pitcher_name"])
        game.batters = home_batters + away_batters
        for b in game.batters:
            all_batters_by_id[b.mlbam_id] = b
        slate.append(game)

        venue_note = "" if is_known_venue(game.park) else \
            "  *** VENUE NOT IN park_factors_live.json — check for a name mismatch/rename, " \
            "currently scoring as neutral ***"
        print(f"  {game.game_id()} @ {game.park} — {len(game.batters)} batters loaded, "
              f"wind={speed}mph dir={wind_dir}, temp={temp}, dome={is_dome}, "
              f"pitchers: {g['away_pitcher_name']} vs {g['home_pitcher_name']}{venue_note}")

    print()
    ranked = rank_slate(slate)  # Stage 1: fast base score for everyone

    apply_matchup_layer(ranked, all_batters_by_id, top_k=TOP_K_FOR_MATCHUP)  # Stage 2: whole slate by default now

    ranked = rank_slate(slate)  # re-score now that matchup_adjustment is filled in on the top K

    parlays = build_parlays(ranked, n_parlays=3, legs=3)

    value_plays = rank_value_plays(ranked, exclude_top_n=20, min_power_score=0.25, top_n=15)
    value_parlays = build_parlays(value_plays, n_parlays=2, legs=3)

    print_report(ranked, parlays, value_plays, value_parlays)

    log_predictions(ranked, target)

    os.makedirs("docs", exist_ok=True)
    html_out = render_html(ranked, parlays, value_plays, value_parlays, target)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"\nWrote mobile-friendly report to docs/index.html ({len(html_out)} chars).")
