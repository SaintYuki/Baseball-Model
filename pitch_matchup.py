"""
pitch_matchup.py — batter's hot pitch type vs pitcher's recently-struggling pitch type

Your example: batter A is crushing sliders in his last ~50 ABs, pitcher A's
slider is getting hit hard in his last 5 starts. This pulls the real
pitch-by-pitch Statcast data for both, breaks it out by pitch type, and
tells you whether that specific overlap is real and how much it should matter
(weighted by how often the pitcher actually throws that pitch).

DATA SOURCE: real pitch-level Statcast data (baseballsavant.mlb.com), via
pybaseball's statcast_batter() / statcast_pitcher(). Same deal as
hr_scanner_auto.py — has to run on your machine, I can't hit Savant from my
sandbox to test this live myself.

KEY METRIC — delta_run_exp: Statcast's run-expectancy change per pitch,
framed from the BATTER'S side:
    positive = good for the batter (a double, a homer, a walk)
    negative = good for the pitcher (a whiff, a weak groundout)
SANITY CHECK THIS FIRST TIME: run it once, glance at a pitch you know the
outcome of (e.g. a pitch that went for a homer), and confirm the sign matches
what you'd expect. I built this off well-documented Statcast conventions but
haven't been able to verify a live pull myself, so don't trust the direction
blind on day one.

STANDALONE FOR NOW — this checks ONE batter vs ONE pitcher at a time on
purpose. Once we confirm the sign convention and the numbers look sane on a
matchup or two you already have a read on, next step is wiring this into
hr_scanner_auto.py so it runs automatically across the whole slate instead of
one at a time.

USAGE:
    python3 pitch_matchup.py <BatterLastName> <BatterFirstName> <PitcherLastName> <PitcherFirstName>
    python3 pitch_matchup.py Judge Aaron Skubal Tarik
"""

import sys
from datetime import date, timedelta

import pandas as pd
import pybaseball as pyb

MIN_PITCHES_FOR_TRUST = 15  # below this per pitch-type sample, we don't trust the number — ignored, not zeroed out


def get_batter_pitch_profile(player_id: int, days_back: int = 21) -> pd.DataFrame:
    """Batter's results by pitch type over the last `days_back` days.
    ~21 days is a rough stand-in for 'last 50 ABs' — tune once you see real volume."""
    end = date.today()
    start = end - timedelta(days=days_back)
    df = pyb.statcast_batter(start.isoformat(), end.isoformat(), player_id)
    if df.empty:
        return pd.DataFrame(columns=["pitch_type", "pitches", "avg_run_value"])
    grouped = (
        df.groupby("pitch_type")
        .agg(pitches=("pitch_type", "count"), avg_run_value=("delta_run_exp", "mean"))
        .reset_index()
    )
    return grouped[grouped["pitches"] >= MIN_PITCHES_FOR_TRUST]


def get_pitcher_pitch_profile(player_id: int, last_n_starts: int = 5) -> pd.DataFrame:
    """Pitcher's results by pitch type over roughly his last `last_n_starts` outings.
    Pulls a wide 45-day window first since we don't know his exact start dates,
    then trims down to the most recent N actual game_pks."""
    end = date.today()
    start = end - timedelta(days=45)
    df = pyb.statcast_pitcher(start.isoformat(), end.isoformat(), player_id)
    if df.empty:
        return pd.DataFrame(columns=["pitch_type", "pitches", "avg_run_value", "usage_pct"])
    recent_games = sorted(df["game_pk"].unique())[-last_n_starts:]
    df = df[df["game_pk"].isin(recent_games)]
    total_pitches = len(df)
    grouped = (
        df.groupby("pitch_type")
        .agg(pitches=("pitch_type", "count"), avg_run_value=("delta_run_exp", "mean"))
        .reset_index()
    )
    grouped["usage_pct"] = grouped["pitches"] / total_pitches
    return grouped[grouped["pitches"] >= MIN_PITCHES_FOR_TRUST]


def matchup_score(batter_profile: pd.DataFrame, pitcher_profile: pd.DataFrame) -> dict:
    """Positive = batter's hot pitch types overlap with pitcher's recently-weak
    pitch types, weighted by how often the pitcher actually throws each one."""
    merged = pd.merge(batter_profile, pitcher_profile, on="pitch_type", suffixes=("_batter", "_pitcher"))
    if merged.empty:
        return {"adjustment": 0.0, "detail": []}
    merged["combined"] = (merged["avg_run_value_batter"] + merged["avg_run_value_pitcher"]) * merged["usage_pct"]
    adjustment = merged["combined"].sum()
    detail = merged[["pitch_type", "usage_pct", "avg_run_value_batter", "avg_run_value_pitcher"]].to_dict("records")
    return {"adjustment": round(float(adjustment), 4), "detail": detail}


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 pitch_matchup.py <BatterLast> <BatterFirst> <PitcherLast> <PitcherFirst>")
        print("Example: python3 pitch_matchup.py Judge Aaron Skubal Tarik")
        sys.exit(1)

    b_last, b_first, p_last, p_first = sys.argv[1:5]

    b_lookup = pyb.playerid_lookup(b_last, b_first)
    p_lookup = pyb.playerid_lookup(p_last, p_first)
    if b_lookup.empty or p_lookup.empty:
        print("Couldn't find one of these players in the lookup — check spelling.")
        sys.exit(1)

    b_id = int(b_lookup.iloc[0]["key_mlbam"])
    p_id = int(p_lookup.iloc[0]["key_mlbam"])

    print(f"Pulling {b_first} {b_last}'s last 21 days by pitch type...")
    b_profile = get_batter_pitch_profile(b_id)
    print(b_profile if not b_profile.empty else "  (no qualifying pitch types — too few pitches seen, or name/date issue)")

    print(f"\nPulling {p_first} {p_last}'s last 5 starts by pitch type...")
    p_profile = get_pitcher_pitch_profile(p_id)
    print(p_profile if not p_profile.empty else "  (no qualifying pitch types)")

    result = matchup_score(b_profile, p_profile)
    print(f"\nMatchup adjustment: {result['adjustment']}")
    for row in result["detail"]:
        print(f"  {row['pitch_type']}: pitcher throws it {row['usage_pct']*100:.0f}% of the time | "
              f"batter avg run value = {row['avg_run_value_batter']:.3f} | "
              f"pitcher avg run value allowed = {row['avg_run_value_pitcher']:.3f}")
