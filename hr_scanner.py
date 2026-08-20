"""
hr_scanner.py — personal Home Run probability scanner (v3)

WHAT'S NEW IN V3
  - PARK_FACTORS is no longer hand-typed. It's loaded at import time from
    data/park_factors_live.json, produced by calibrate_model.py +
    export_live_factors.py. Every value in that file already survived a
    logistic regression controlling for weather, team quality, and batter
    power AND a significance test — a park/hand combo that didn't clear
    that bar is 0.0 (neutral) in the file, not a guess.
  - weather_adjustment is no longer a crude "wind speed + in/out" heuristic.
    It looks up the same calibrated table (data/weather_factors_live.json)
    used for park factors, keyed by (batter handedness, temp bucket, wind
    direction, wind speed bucket) — the exact same buckets calibrate_weather.py
    used, via bucket_temp/bucket_speed/categorize_wind imported from there.
    This means weather now depends on the BATTER'S handedness, same as park
    does — a game no longer has one single weather_adj, each batter does.

HOW THE SCORE WORKS
    base   = power_score * (1 + park_factor) * (1 + weather_adj) * (1 + platoon_bonus)
    final  = base * (1 + clamp(matchup_adjustment * MATCHUP_SCALE, -0.25, 0.25))

  power_score        : 0.0-1.0 recent power proxy (shrunk barrel rate from
                        hr_scanner_auto.py, or fill by hand — see below).
  park_factor         : calibrated HR boost/suppression for park + batter
                        handedness. 0.0 if the park file doesn't have this
                        venue (see is_known_venue) or the effect wasn't
                        significant.
  weather_adj         : calibrated HR boost/suppression for this batter's
                        handedness under these exact temp/wind conditions.
                        0.0 for domes, missing temp/wind, or any bucket that
                        wasn't significant (or wasn't observed at all) in
                        the calibration data.
  platoon_bonus       : optional nudge for strong/weak platoon splits.
  matchup_adjustment  : from pitch_matchup.py — batter's recent form vs a
                        pitch type overlapped with the pitcher's recent
                        struggles on that same pitch type, weighted by how
                        often the pitcher throws it. 0.0 if not computed
                        (e.g. batter didn't make the top-K cut for the
                        expensive per-player pull — see hr_scanner_auto.py).
                        Scaled by MATCHUP_SCALE and clamped so one noisy
                        pitch-type read can't swing the score more than a
                        strong park factor would.

  - New: rank_value_plays() surfaces "under the radar" players — real
    situational lift (park+weather+matchup) stacked on their own power
    score, specifically excluding whoever's already in the overall top
    ranks. The overall #1 ranked bat is almost always famous and already
    priced into the sportsbook's line; a moderate-tier bat getting an
    unusual boost today is a better place to look for an edge.

VENUE NAME RISK: park_factors_live.json's keys came from historical game
feeds during the calibration pull. If a park's official name changed
between then and now (Daikin Park, loanDepot park, Rate Field, and UNIQLO
Field at Dodger Stadium have all been renamed in recent memory), a live
game using the new name will silently miss the lookup and fall back to
neutral — not a crash, just quietly treating a real park as unknown. Use
is_known_venue() to check instead of assuming a 0.0 means "not significant."

MANUAL USE (no auto-pull): edit the SLATE block at the bottom and run
    python3 hr_scanner.py
AUTOMATED USE: run hr_scanner_auto.py instead, which pulls everything live
and calls into this file's scoring engine.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from calibrate_weather import bucket_speed, bucket_temp

MATCHUP_SCALE = 3.0
MATCHUP_CLAMP = 0.25  # max +/- swing from the matchup layer, same order of magnitude as a strong park factor

# A single significant park or weather factor can otherwise dominate a whole
# night's ranking — e.g. Tropicana Field R showed +79% one day, which alone
# pushed every qualifying Rays/Orioles batter to the top of the board
# regardless of their actual relative form. Clamping keeps any ONE
# situational factor (park or weather) from swinging a score more than a
# strong matchup edge could — same order of magnitude as MATCHUP_CLAMP,
# deliberately, so no single input category can dominate the others.
SITUATIONAL_FACTOR_CLAMP = 0.25

PARK_FACTORS_PATH = "data/park_factors_live.json"
WEATHER_FACTORS_PATH = "data/weather_factors_live.json"

DEFAULT_PARK_FACTOR = {"L": 0.0, "R": 0.0}


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _load_calibrated_json(path: str, label: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"[hr_scanner] loaded calibrated {label} from {path}")
        return data
    except FileNotFoundError:
        print(f"[hr_scanner] WARNING: {path} not found — run calibrate_model.py then "
              f"export_live_factors.py first. Using neutral (0.0) {label} for everything "
              f"until that file exists.")
        return {}
    except json.JSONDecodeError as e:
        print(f"[hr_scanner] WARNING: {path} exists but isn't valid JSON ({e}) — "
              f"using neutral (0.0) {label} for everything. Re-run export_live_factors.py.")
        return {}


# ---------------------------------------------------------------------------
# 1. PARK FACTORS + WEATHER FACTORS — loaded once at import time from the
#    calibration pipeline's output. See export_live_factors.py for the rule
#    (only statistically significant effects survive; everything else is 0.0).
# ---------------------------------------------------------------------------
PARK_FACTORS = _load_calibrated_json(PARK_FACTORS_PATH, "park factors")
WEATHER_FACTORS = _load_calibrated_json(WEATHER_FACTORS_PATH, "weather factors")


def is_known_venue(park: str) -> bool:
    """True if this exact venue string exists as a key in the calibrated
    park factors file. False means either the file wasn't loaded, or this
    venue's name doesn't match anything the calibration saw — worth checking
    for a rename before assuming the park really has no effect."""
    return park in PARK_FACTORS


def get_park_factor(park: str, stand: str) -> float:
    raw = PARK_FACTORS.get(park, DEFAULT_PARK_FACTOR).get(stand, 0.0)
    return _clamp(raw, SITUATIONAL_FACTOR_CLAMP)


@dataclass
class Batter:
    name: str
    team: str
    bats: str                        # "L" or "R"
    power_score: float               # 0.0-1.0 recent power proxy
    platoon_bonus: float = 0.0
    mlbam_id: Optional[int] = None           # needed for the pitch-matchup pull
    opp_pitcher_id: Optional[int] = None     # opposing starter's MLBAM id
    opp_pitcher_name: str = ""
    matchup_adjustment: float = 0.0          # raw value from pitch_matchup.py, filled in later


@dataclass
class Game:
    home_team: str
    away_team: str
    park: str
    is_dome: bool
    wind_speed_mph: float = 0.0
    wind_dir: Optional[str] = None   # Calm / Out / In / Cross / Varies / Unknown — see calibrate_weather.categorize_wind
    temp: Optional[float] = None     # degrees F, from the live/historical weather feed
    home_pitcher_id: Optional[int] = None
    home_pitcher_name: str = ""
    away_pitcher_id: Optional[int] = None
    away_pitcher_name: str = ""
    game_pk: Optional[int] = None    # MLB game id — lets check_results.py tell "this game was postponed" from "this game happened, no HR"
    batters: list = field(default_factory=list)

    def game_id(self) -> str:
        return f"{self.away_team}@{self.home_team}"


def weather_adjustment(game: Game, stand: str) -> float:
    """Calibrated weather effect for a batter of this handedness under this
    game's exact temp/wind bucket. Neutral (0.0) for domes, missing
    temp/wind data, or any bucket the calibration didn't find significant
    (or never observed at all — a bucket simply absent from the JSON behaves
    the same as one that was tested and came back neutral). Clamped for the
    same reason park factor is — a rare significant weather bucket shouldn't
    be able to swing a score any more than a strong matchup edge could."""
    if game.is_dome or game.temp is None or not game.wind_dir:
        return 0.0
    temp_bucket = bucket_temp(game.temp)
    speed_bucket = bucket_speed(game.wind_speed_mph)
    raw = (
        WEATHER_FACTORS.get(stand, {})
        .get(temp_bucket, {})
        .get(game.wind_dir, {})
        .get(speed_bucket, 0.0)
    )
    return _clamp(raw, SITUATIONAL_FACTOR_CLAMP)


def score_batter(b: Batter, g: Game) -> float:
    park = get_park_factor(g.park, b.bats)
    weather = weather_adjustment(g, b.bats)
    base = b.power_score * (1 + park) * (1 + weather) * (1 + b.platoon_bonus)
    matchup_mult = 1 + _clamp(b.matchup_adjustment * MATCHUP_SCALE, MATCHUP_CLAMP)
    return base * matchup_mult


def rank_slate(games: list[Game]) -> list[dict]:
    rows = []
    for g in games:
        for b in g.batters:
            park = get_park_factor(g.park, b.bats)
            wx = weather_adjustment(g, b.bats)
            score = round(score_batter(b, g), 4)
            # situational_boost isolates how much park+weather+matchup moved this player's
            # score relative to their OWN power_score alone — i.e. the part of the score
            # that's about today's specific circumstances, not about how good the hitter is
            # in general. That second part is exactly what the sportsbook already knows and
            # prices into the odds; the situational part is the more likely place to find an
            # edge, especially on a player nobody's watching closely.
            situational_boost = round(score / b.power_score - 1, 4) if b.power_score else 0.0
            rows.append({
                "player": b.name,
                "mlbam_id": b.mlbam_id,
                "team": b.team,
                "game": g.game_id(),
                "game_pk": g.game_pk,
                "park": g.park,
                "park_factor": park,
                "weather_adj": wx,
                "matchup_adjustment": b.matchup_adjustment,
                "opp_pitcher": b.opp_pitcher_name,
                "power_score": b.power_score,
                "situational_boost": situational_boost,
                "score": score,
            })
    rows.sort(key=lambda r: -r["score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def rank_value_plays(ranked: list[dict], exclude_top_n: int = 20,
                      min_power_score: float = 0.25, top_n: int = 15,
                      max_per_park: int = 2) -> list[dict]:
    """'Under the radar' plays: real situational lift (park + weather +
    matchup, specific to today) stacked on top of a real underlying power
    bat — deliberately excluding whoever's already sitting in the overall
    top N, since those are exactly the names a sportsbook has already priced
    accordingly. min_power_score keeps this from surfacing a bench bat with
    a trivial power score just because it's getting multiplied by a big
    park factor — a +40% boost on a 0.10 power score is still a low
    absolute shot, not a real value play, just a small number times a
    bigger number.

    max_per_park caps how many slots a single park can consume. A park
    factor applies identically to every qualifying batter who plays there
    that night — without this cap, one big (even legitimately significant)
    park factor can fill most of the board with what's really one insight
    wearing a dozen different names, crowding out genuinely distinct
    matchup-driven reads elsewhere on the slate."""
    candidates = [r for r in ranked if r["rank"] > exclude_top_n and r["power_score"] >= min_power_score]
    candidates = sorted(candidates, key=lambda r: -r["situational_boost"])

    selected = []
    park_counts: dict[str, int] = {}
    for r in candidates:
        park = r["park"]
        if park_counts.get(park, 0) >= max_per_park:
            continue
        selected.append(r)
        park_counts[park] = park_counts.get(park, 0) + 1
        if len(selected) == top_n:
            break
    return selected


def build_parlays(ranked: list[dict], n_parlays: int = 3, legs: int = 3) -> list[list[dict]]:
    """Name alone isn't a safe identity — MLB has more than one active player
    named Max Muncy, for instance. Dedup on mlbam_id when the row has one
    (real auto-pulled slates always do); fall back to name only for rows
    without an id (e.g. hand-built manual SLATE entries in demo mode)."""
    def player_key(r):
        return r["mlbam_id"] if r.get("mlbam_id") is not None else r["player"]

    used_players = set()
    parlays = []
    for _ in range(n_parlays):
        leg_games = set()
        parlay = []
        for r in ranked:
            if player_key(r) in used_players or r["game"] in leg_games:
                continue
            parlay.append(r)
            leg_games.add(r["game"])
            used_players.add(player_key(r))
            if len(parlay) == legs:
                break
        if len(parlay) == legs:
            parlays.append(parlay)
    return parlays


def print_report(ranked: list[dict], parlays: list[list[dict]],
                  value_plays: Optional[list[dict]] = None,
                  value_parlays: Optional[list[list[dict]]] = None):
    print("=" * 88)
    print("FULL RANKED SLATE")
    print("=" * 88)
    for r in ranked:
        wx_note = f"wx {'+' if r['weather_adj']>0 else ''}{r['weather_adj']*100:.1f}%" if r["weather_adj"] else "wx neutral"
        matchup_note = f"matchup {r['matchup_adjustment']:+.3f}" if r["matchup_adjustment"] else "matchup n/a"
        print(f"#{r['rank']:>3}  {r['player']:<22} {r['team']:<4} {r['game']:<14} "
              f"park {r['park_factor']*100:+.0f}%  {wx_note:<14} {matchup_note:<16} score {r['score']:.3f}")

    print()
    print("=" * 88)
    print("PARLAY LAYS (no two legs same game, no player repeated across lays)")
    print("=" * 88)
    for i, p in enumerate(parlays, 1):
        print(f"\nLay {i}:")
        for leg in p:
            print(f"  - {leg['player']:<22} ({leg['team']}, {leg['game']}) vs {leg['opp_pitcher']:<20} "
                  f"score {leg['score']:.3f}  rank #{leg['rank']}")

    if value_plays:
        print()
        print("=" * 88)
        print("VALUE BOARD — under-the-radar plays: real situational lift, outside the overall top ranks")
        print("=" * 88)
        for r in value_plays:
            wx_display = f"wx {r['weather_adj']*100:+.1f}%, " if r["weather_adj"] else ""
            print(f"  {r['player']:<22} {r['team']:<4} {r['game']:<14} rank #{r['rank']:<4} "
                  f"power {r['power_score']:.3f}  situational boost {r['situational_boost']*100:+.1f}%  "
                  f"(park {r['park_factor']*100:+.0f}%, {wx_display}matchup {r['matchup_adjustment']:+.3f})  "
                  f"score {r['score']:.3f}")

    if value_parlays:
        print()
        print("=" * 88)
        print("VALUE PARLAY LAYS (built from the Value Board, not the overall ranking)")
        print("=" * 88)
        for i, p in enumerate(value_parlays, 1):
            print(f"\nValue Lay {i}:")
            for leg in p:
                print(f"  - {leg['player']:<22} ({leg['team']}, {leg['game']}) vs {leg['opp_pitcher']:<20} "
                      f"boost {leg['situational_boost']*100:+.1f}%  score {leg['score']:.3f}  rank #{leg['rank']}")


# ---------------------------------------------------------------------------
# 2. MANUAL SLATE — for hand-editing without the live auto-pull. Sample only.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SLATE = [
        Game(
            home_team="COL", away_team="KC", park="Coors Field", is_dome=False,
            wind_speed_mph=7, wind_dir="Out", temp=78,
            batters=[
                Batter("Sample Rockies Bat 1", "COL", "R", power_score=0.55),
                Batter("Sample Royals Bat 1", "KC", "L", power_score=0.48),
            ],
        ),
    ]
    ranked = rank_slate(SLATE)
    parlays = build_parlays(ranked, n_parlays=3, legs=3)
    value_plays = rank_value_plays(ranked, exclude_top_n=0)  # exclude_top_n=0 since this demo slate is tiny
    print_report(ranked, parlays, value_plays)
