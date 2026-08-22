"""
tb_scanner.py — Path 2, prop 2: the total-bases version of hr_scanner.py.
Same scoring shape, same clamp discipline, same Value Board idea — just
targeting "2+ total bases in a game" instead of "1+ HR", reading from
data/tb_park_factors_live.json / data/tb_weather_factors_live.json
(produced by calibrate_tb_model.py + export_tb_live_factors.py).

WHAT'S REUSED FROM hr_scanner.py, NOT DUPLICATED: _clamp, the
SITUATIONAL_FACTOR_CLAMP / MATCHUP_SCALE / MATCHUP_CLAMP constants, the
Game dataclass, and build_parlays — same genuinely prop-agnostic pieces
hits_scanner.py already reuses.

WHAT'S NEW HERE: PARK_FACTORS/WEATHER_FACTORS point at the TB-specific JSON
files (own significance history, own stability gate, per
export_tb_live_factors.py). The Batter dataclass carries tb_rate instead of
power_score/hit_rate — same "don't share the field name across props"
reasoning as hits_scanner.py's Batter class. Labeled "slug" in the printed
report (short for slugging, the natural baseball term for this quantity),
same brevity as "power"/"contact" elsewhere.

ONE THING WORTH REMEMBERING FROM CALIBRATION THAT DOESN'T CHANGE ANYTHING
HERE: the *outcome* calibrate_tb_model.py targets (is_2plus_tb) is a
whole-game fact — two singles satisfy it just like one double does. That
aggregation already happened upstream, at calibration time. This file's
scoring is still a per-batter, per-game score exactly like every other
prop; nothing about that game-level aggregation needs to be re-done here.

HOW THE SCORE WORKS (same shape as every other prop)
    base   = tb_rate * (1 + park_factor) * (1 + weather_adj) * (1 + platoon_bonus)
    final  = base * (1 + clamp(matchup_adjustment * MATCHUP_SCALE, -0.25, 0.25))

  tb_rate : trailing total-bases-per-batted-ball, NOT rescaled onto a 0-1
            "eliteness" dial (same choice hits_scanner.py made, for the
            same reason — real values already sit in an interpretable
            range without compression). What exact source feeds this is a
            decision for tb_scanner_auto.py (not yet built) — this module
            just scores whatever tb_rate a Batter carries.

MANUAL USE (no auto-pull): edit the SLATE block at the bottom and run
    python3 tb_scanner.py
AUTOMATED USE: tb_scanner_auto.py (not yet built) would pull live data the
same way hits_scanner_auto.py does and call into this file's engine.
"""

import json
from typing import Optional

from calibrate_weather import bucket_speed, bucket_temp
from hr_scanner import (
    _clamp, SITUATIONAL_FACTOR_CLAMP, MATCHUP_SCALE, MATCHUP_CLAMP,
    Game, build_parlays,
)

PARK_FACTORS_PATH = "data/tb_park_factors_live.json"
WEATHER_FACTORS_PATH = "data/tb_weather_factors_live.json"

DEFAULT_PARK_FACTOR = {"L": 0.0, "R": 0.0}


def _load_calibrated_json(path: str, label: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"[tb_scanner] loaded calibrated {label} from {path}")
        return data
    except FileNotFoundError:
        print(f"[tb_scanner] WARNING: {path} not found — run calibrate_tb_model.py then "
              f"export_tb_live_factors.py first. Using neutral (0.0) {label} for everything "
              f"until that file exists.")
        return {}
    except json.JSONDecodeError as e:
        print(f"[tb_scanner] WARNING: {path} exists but isn't valid JSON ({e}) — "
              f"using neutral (0.0) {label} for everything. Re-run export_tb_live_factors.py.")
        return {}


PARK_FACTORS = _load_calibrated_json(PARK_FACTORS_PATH, "TB park factors")
WEATHER_FACTORS = _load_calibrated_json(WEATHER_FACTORS_PATH, "TB weather factors")


def is_known_venue(park: str) -> bool:
    return park in PARK_FACTORS


def get_park_factor(park: str, stand: str) -> float:
    raw = PARK_FACTORS.get(park, DEFAULT_PARK_FACTOR).get(stand, 0.0)
    return _clamp(raw, SITUATIONAL_FACTOR_CLAMP)


def weather_adjustment(game: Game, stand: str) -> float:
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


class Batter:
    """Deliberately not shared with hr_scanner.Batter or hits_scanner.Batter —
    tb_rate, hit_rate, and power_score are different quantities on different
    natural scales, and giving them the same field name across props risks
    silently mixing them up somewhere downstream. Same fields otherwise."""
    def __init__(self, name: str, team: str, bats: str, tb_rate: float,
                 platoon_bonus: float = 0.0, mlbam_id: Optional[int] = None,
                 opp_pitcher_id: Optional[int] = None, opp_pitcher_name: str = "",
                 matchup_adjustment: float = 0.0):
        self.name = name
        self.team = team
        self.bats = bats
        self.tb_rate = tb_rate
        self.platoon_bonus = platoon_bonus
        self.mlbam_id = mlbam_id
        self.opp_pitcher_id = opp_pitcher_id
        self.opp_pitcher_name = opp_pitcher_name
        self.matchup_adjustment = matchup_adjustment


def score_batter(b: Batter, g: Game) -> float:
    park = get_park_factor(g.park, b.bats)
    weather = weather_adjustment(g, b.bats)
    base = b.tb_rate * (1 + park) * (1 + weather) * (1 + b.platoon_bonus)
    matchup_mult = 1 + _clamp(b.matchup_adjustment * MATCHUP_SCALE, MATCHUP_CLAMP)
    return base * matchup_mult


def rank_slate(games: list[Game]) -> list[dict]:
    rows = []
    for g in games:
        for b in g.batters:
            park = get_park_factor(g.park, b.bats)
            wx = weather_adjustment(g, b.bats)
            score = round(score_batter(b, g), 4)
            situational_boost = round(score / b.tb_rate - 1, 4) if b.tb_rate else 0.0
            rows.append({
                "player": b.name, "mlbam_id": b.mlbam_id, "team": b.team,
                "game": g.game_id(), "game_pk": g.game_pk, "park": g.park,
                "park_factor": park, "weather_adj": wx,
                "matchup_adjustment": b.matchup_adjustment,
                "opp_pitcher": b.opp_pitcher_name,
                "tb_rate": b.tb_rate, "situational_boost": situational_boost,
                "score": score,
            })
    rows.sort(key=lambda r: -r["score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def rank_value_plays(ranked: list[dict], exclude_top_n: int = 20,
                      min_tb_rate: float = 0.3, top_n: int = 15,
                      max_per_park: int = 2) -> list[dict]:
    """Same idea as hits_scanner's version: real situational lift stacked
    on a real underlying rate, excluding the obvious top names, capped per
    park so one big park factor can't fill the whole board. min_tb_rate
    default of 0.3 is set from the real calibration's own p10 (~0.34 median
    across hands) — a rough floor for 'not a truly cold bat', same idea as
    hits_scanner's 0.25 default relative to its own real p10."""
    candidates = [r for r in ranked if r["rank"] > exclude_top_n and r["tb_rate"] >= min_tb_rate]
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


def print_report(ranked: list[dict], parlays: list[list[dict]],
                  value_plays: Optional[list[dict]] = None,
                  value_parlays: Optional[list[list[dict]]] = None):
    print("=" * 88)
    print("FULL RANKED SLATE (TOTAL BASES, 2+)")
    print("=" * 88)
    for r in ranked:
        wx_note = f"wx {'+' if r['weather_adj']>0 else ''}{r['weather_adj']*100:.1f}%" if r["weather_adj"] else "wx neutral"
        matchup_note = f"matchup {r['matchup_adjustment']:+.3f}" if r["matchup_adjustment"] else "matchup n/a"
        print(f"#{r['rank']:>3}  {r['player']:<22} {r['team']:<4} {r['game']:<14} "
              f"park {r['park_factor']*100:+.0f}%  {wx_note:<14} {matchup_note:<16} score {r['score']:.3f}")

    print()
    print("=" * 88)
    print("TOTAL BASES PARLAY LAYS (no two legs same game, no player repeated across lays)")
    print("=" * 88)
    for i, p in enumerate(parlays, 1):
        print(f"\nLay {i}:")
        for leg in p:
            print(f"  - {leg['player']:<22} ({leg['team']}, {leg['game']}) vs {leg['opp_pitcher']:<20} "
                  f"score {leg['score']:.3f}  rank #{leg['rank']}")

    if value_plays:
        print()
        print("=" * 88)
        print("TOTAL BASES VALUE BOARD — under-the-radar plays: real situational lift, outside the overall top ranks")
        print("=" * 88)
        for r in value_plays:
            wx_display = f"wx {r['weather_adj']*100:+.1f}%, " if r["weather_adj"] else ""
            print(f"  {r['player']:<22} {r['team']:<4} {r['game']:<14} rank #{r['rank']:<4} "
                  f"slug {r['tb_rate']:.3f}  situational boost {r['situational_boost']*100:+.1f}%  "
                  f"(park {r['park_factor']*100:+.0f}%, {wx_display}matchup {r['matchup_adjustment']:+.3f})  "
                  f"score {r['score']:.3f}")

    if value_parlays:
        print()
        print("=" * 88)
        print("TOTAL BASES VALUE PARLAY LAYS (built from the Value Board, not the overall ranking)")
        print("=" * 88)
        for i, p in enumerate(value_parlays, 1):
            print(f"\nValue Lay {i}:")
            for leg in p:
                print(f"  - {leg['player']:<22} ({leg['team']}, {leg['game']}) vs {leg['opp_pitcher']:<20} "
                      f"boost {leg['situational_boost']*100:+.1f}%  score {leg['score']:.3f}  rank #{leg['rank']}")


if __name__ == "__main__":
    SLATE = [
        Game(
            home_team="COL", away_team="KC", park="Coors Field", is_dome=False,
            wind_speed_mph=7, wind_dir="Out", temp=78,
            batters=[
                Batter("Sample Rockies Bat 1", "COL", "L", tb_rate=0.55),
                Batter("Sample Royals Bat 1", "KC", "R", tb_rate=0.42),
            ],
        ),
    ]
    ranked = rank_slate(SLATE)
    parlays = build_parlays(ranked, n_parlays=3, legs=3)
    value_plays = rank_value_plays(ranked, exclude_top_n=0)
    print_report(ranked, parlays, value_plays)
