"""
mlb_live_data.py — schedule, weather, handedness, and matchup-pull functions
shared between hr_scanner_auto.py and hits_scanner_auto.py. Extracted out of
hr_scanner_auto.py specifically to let hits_scanner_auto.py import this
shared logic without creating a circular import: once hr_scanner_auto.py
also needed to import hits-specific functions (to build the combined HR+hits
report), hits_scanner_auto.py importing directly from hr_scanner_auto.py
would have meant each module trying to import the other, which Python can't
resolve. None of the functions here were ever HR-specific in the first
place — schedule/weather/handedness pulls, and the pitch-level matchup
layer, don't care what prop is being scored.

SETUP: pip install pybaseball MLB-StatsAPI pandas
"""

import time
from typing import Optional

import statsapi

from pitch_matchup import get_batter_pitch_profile, get_pitcher_pitch_profile, matchup_score

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
    dict on failure so callers degrade to the 'R' default rather than crash."""
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


def apply_matchup_layer(ranked: list[dict], all_batters_by_id: dict,
                         top_k: Optional[int] = TOP_K_FOR_MATCHUP):
    """Mutates matchup_adjustment on the top_k Batter objects in place (or
    on everyone, if top_k is None), using real pitch-by-pitch data. Pitcher
    profiles are cached so a starter facing 13 opposing batters only gets
    pulled once, not 13 times — this cache is what keeps an uncapped run
    from scaling anywhere near as badly as "one call per batter" sounds;
    it's really "one call per batter, plus ~15-30 calls total for the
    day's starters," not multiplicatively worse per batter added.

    Works identically for HR or hits Batter objects — both carry mlbam_id,
    opp_pitcher_id, opp_pitcher_name, and matchup_adjustment, and this
    function never touches power_score or hit_rate specifically.

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
