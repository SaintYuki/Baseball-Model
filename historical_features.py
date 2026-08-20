"""
historical_features.py — point-in-time engineered features for Phase 4b.

WHY A SEPARATE FILE: calibrate_model.py fits the regression; this file builds
the features that go into it. Batter power, and now the pitch-type matchup
signal — keeping them here means both get built and tested the same way
without bloating the model-fitting code.

CRITICAL DESIGN RULE — point-in-time correctness: every feature here uses
ONLY data strictly BEFORE the date it's describing (pandas rolling with
closed='left'). A batter's power score on any given date must never include
that date's own outcome, or the model would be predicting home runs partly
using the home run itself — this was tested explicitly (see historical_data
conversation) before shipping.

PITCH-TYPE MATCHUP — needs a DIFFERENT pull than batter_power: historical_data.py
only saved balls in play (type == 'X'), but a real matchup signal needs every
pitch — whiffs, called strikes, fouls — same as pitch_matchup.py does live.
pull_full_pitch_data() re-calls pybaseball.statcast() for the same date range,
which should hit pybaseball's own cache (enabled below) instead of re-downloading,
since historical_data.py already triggered that exact call internally before
filtering down to batted balls only.

DESIGN CHOICE: rather than replicate pitch_matchup.py's hand-crafted formula
(batter run value + pitcher run value allowed, weighted by usage%), this feeds
the batter's trailing run value against a pitch type AND the pitcher's trailing
run value allowed on that pitch type into the regression as two separate
continuous features, and lets the model itself learn how to weight them.

RUN THIS FIRST, BY ITSELF, before anything else touches it:
    python3 historical_features.py
It'll load your cached batted-ball data and report whether the required
columns (batter, game_date) are present, then show a few real examples so
you can eyeball whether the numbers look sane before we wire this into the
regression.
"""

import glob
import sys

import pandas as pd
import pybaseball as pyb
import pybaseball.cache as pyb_cache

pyb_cache.enable()

BATTER_POWER_WINDOW_DAYS = 21
MIN_PRIOR_BATTED_BALLS = 30  # need at least this many prior batted balls to trust a batter's trailing rate

# --- Path 2 pilot: hits prop ---
# Separate constants from the HR ones above (even though they start at the same values) so
# tuning one prop's window later can never silently affect another's. Hits happen roughly
# 10x more often than HRs per batted ball (real MLB BABIP is ~0.29-0.30 vs ~0.03 HR rate),
# so this window is probably more conservative than strictly necessary at the same threshold
# (see the standard-error math: same n=30 gives ~28% relative uncertainty for hit rate vs
# ~104% for HR rate) — kept equal to the HR window to start, for consistency; revisit once
# we've seen the real trailing-rate distribution, same as TRAILING_HR_RATE_ELITE was revised
# after seeing real numbers rather than guessed up front.
HIT_RATE_WINDOW_DAYS = 21
MIN_PRIOR_BATTED_BALLS_FOR_HITS = 30
HIT_EVENTS = {"single", "double", "triple", "home_run"}  # Statcast's events column values for
                                                          # a hit — worth a live sanity check
                                                          # the first time this runs for real,
                                                          # same as every other new integration.

PITCH_MATCHUP_WINDOW_DAYS = 21
MIN_PITCHES_FOR_TRUST = 15  # per (player, pitch_type) — same threshold as live pitch_matchup.py


def load_batted_balls(path_arg: str | None) -> pd.DataFrame:
    if path_arg:
        return pd.read_parquet(path_arg)
    files = glob.glob("data/batted_balls_*.parquet")
    if not files:
        print("No cached batted-ball files found in ./data/ — run historical_data.py first.")
        sys.exit(1)
    print(f"Loading {len(files)} cached file(s): {files}")
    combined = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # date ranges pulled at different times can overlap (e.g. an early pilot pull whose
    # range is fully contained in a later full-season pull) — glob-and-concat alone
    # double-counts every pitch in the overlap. game_pk+at_bat_number+pitch_number
    # uniquely identifies a single pitch in Statcast data, so dedup on that.
    before = len(combined)
    combined = combined.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
    if before != len(combined):
        print(f"Removed {before - len(combined)} duplicate rows from overlapping cached files.")
    return combined


def add_batter_power(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time trailing HR rate per batter — 21-day rolling window,
    strictly excluding the current day. Batters without at least
    MIN_PRIOR_BATTED_BALLS prior batted balls get NaN, not a misleadingly
    confident early-season number off a handful of at-bats."""
    required = {"batter", "game_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s) {missing} for point-in-time batter power. "
            f"Available columns: {sorted(df.columns)[:30]}..."
        )

    df = df.copy()
    df["is_hr"] = (df["events"] == "home_run").astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["batter", "game_date"])

    out = []
    for batter_id, g in df.groupby("batter"):
        g = g.set_index("game_date")
        n_prior = g["is_hr"].rolling(f"{BATTER_POWER_WINDOW_DAYS}D", closed="left").count()
        rate = g["is_hr"].rolling(f"{BATTER_POWER_WINDOW_DAYS}D", closed="left").mean()
        rate = rate.where(n_prior >= MIN_PRIOR_BATTED_BALLS)
        g = g.reset_index()
        g["batter_power"] = rate.values
        g["batter_power_n"] = n_prior.values
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _trailing_by_group(df: pd.DataFrame, group_cols: list[str], value_col: str,
                        window_days: int, min_n: int, out_col: str,
                        return_count: bool = False) -> pd.DataFrame:
    """Shared rolling-trailing-average machinery, reused for both the batter
    side and pitcher side of the pitch-type matchup signal, and now the
    trailing hit-rate feature too. Same closed='left' discipline as
    add_batter_power — never includes the current day. return_count=True
    also attaches f'{out_col}_n' (how many prior observations backed the
    estimate), useful for coverage diagnostics — off by default so the
    existing matchup callers are completely unaffected."""
    df = df.sort_values(group_cols + ["game_date"])
    out = []
    for _, g in df.groupby(group_cols):
        g = g.set_index("game_date")
        n_prior = g[value_col].rolling(f"{window_days}D", closed="left").count()
        trailing = g[value_col].rolling(f"{window_days}D", closed="left").mean()
        trailing = trailing.where(n_prior >= min_n)
        g = g.reset_index()
        g[out_col] = trailing.values
        if return_count:
            g[f"{out_col}_n"] = n_prior.values
        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_batter_hit_rate(df: pd.DataFrame, window_days: int = HIT_RATE_WINDOW_DAYS,
                         min_prior_batted_balls: int = MIN_PRIOR_BATTED_BALLS_FOR_HITS) -> pd.DataFrame:
    """Point-in-time trailing HIT rate per batter — the Path 2 pilot's
    version of add_batter_power. Same point-in-time discipline (closed='left':
    a batter's own game never leaks into their own trailing value), built on
    the shared _trailing_by_group engine above instead of a second hand-rolled
    copy of the rolling-window loop that add_batter_power uses (that function
    is left untouched rather than refactored onto this engine too — it's
    heavily depended on already, not worth the risk for a DRY-ness win alone).

    Reuses the EXACT SAME source data as add_batter_power
    (data/batted_balls_*.parquet, which only contains type=='X' rows) — a hit,
    like a HR, can only happen via a batted ball, so no new data pull is
    needed for this feature at all."""
    required = {"batter", "game_date", "events"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s) {missing} for point-in-time hit rate. "
            f"Available columns: {sorted(df.columns)[:30]}..."
        )

    df = df.copy()
    df["is_hit"] = df["events"].isin(HIT_EVENTS).astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])

    return _trailing_by_group(df, ["batter"], "is_hit", window_days,
                               min_prior_batted_balls, "batter_hit_rate", return_count=True)


def pull_full_pitch_data(start_dt: str, end_dt: str) -> pd.DataFrame:
    """Every pitch, not just balls in play — needed for the matchup signal.
    Should hit pybaseball's cache rather than re-downloading, since
    historical_data.py already made this exact call internally."""
    print(f"Pulling full pitch-level data {start_dt} to {end_dt} (should hit cache if "
          f"historical_data.py already covered this range)...")
    df = pyb.statcast(start_dt, end_dt)
    print(f"  {len(df)} total pitches.")
    return df


def add_pitch_type_matchup(pitches: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Attaches two point-in-time features to `target` (the batted-ball
    modeling dataset): batter_pitch_rv (batter's trailing run value against
    THIS specific pitch type) and pitcher_pitch_rv_allowed (pitcher's trailing
    run value allowed on that same pitch type). Both computed from the FULL
    pitch-level data (pitches), not just contact, then merged onto target by
    matching (player, pitch_type, game_date) — the merge is exact on pitch_type
    because a batted ball only tells us about the ONE pitch that got hit, so
    we want that specific pitch type's trailing signal, not an average
    across all pitch types."""
    required = {"batter", "pitcher", "pitch_type", "game_date", "delta_run_exp"}
    missing = required - set(pitches.columns)
    if missing:
        raise ValueError(f"Missing required column(s) {missing} in pitch-level data.")

    pitches = pitches.dropna(subset=["pitch_type", "delta_run_exp"]).copy()
    pitches["game_date"] = pd.to_datetime(pitches["game_date"])

    print("Computing trailing batter-vs-pitch-type run value (this is the slow part)...")
    batter_side = _trailing_by_group(
        pitches, ["batter", "pitch_type"], "delta_run_exp",
        PITCH_MATCHUP_WINDOW_DAYS, MIN_PITCHES_FOR_TRUST, "batter_pitch_rv"
    )
    batter_lookup = (
        batter_side.dropna(subset=["batter_pitch_rv"])
        .drop_duplicates(["batter", "pitch_type", "game_date"])
        [["batter", "pitch_type", "game_date", "batter_pitch_rv"]]
    )

    print("Computing trailing pitcher-vs-pitch-type run value allowed...")
    pitcher_side = _trailing_by_group(
        pitches, ["pitcher", "pitch_type"], "delta_run_exp",
        PITCH_MATCHUP_WINDOW_DAYS, MIN_PITCHES_FOR_TRUST, "pitcher_pitch_rv_allowed"
    )
    pitcher_lookup = (
        pitcher_side.dropna(subset=["pitcher_pitch_rv_allowed"])
        .drop_duplicates(["pitcher", "pitch_type", "game_date"])
        [["pitcher", "pitch_type", "game_date", "pitcher_pitch_rv_allowed"]]
    )

    target = target.copy()
    target["game_date"] = pd.to_datetime(target["game_date"])
    target = target.merge(batter_lookup, on=["batter", "pitch_type", "game_date"], how="left")
    target = target.merge(pitcher_lookup, on=["pitcher", "pitch_type", "game_date"], how="left")
    return target


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    batted = load_batted_balls(path_arg)

    print(f"\nColumns available: {sorted(batted.columns)}")
    print(f"\nHas 'batter' column: {'batter' in batted.columns}")
    print(f"Has 'game_date' column: {'game_date' in batted.columns}")

    result = add_batter_power(batted)

    coverage = result["batter_power"].notna().mean()
    print(f"\n{coverage*100:.0f}% of rows have a usable batter_power value "
          f"(rest are early-season / not enough prior history yet — expected, not a bug).")

    print("\nSample: an EVERYDAY player's power score over time (most batted balls in this dataset,")
    print("so we get a real look at whether the trailing score actually moves game to game):")
    busiest_batter = result["batter"].value_counts().idxmax()
    sample = result[result["batter"] == busiest_batter][["game_date", "batter_power", "batter_power_n"]]
    sample = sample.drop_duplicates("game_date").sort_values("game_date")
    print(f"(batter id {busiest_batter}, {len(sample)} game-dates)")
    print(sample.iloc[::max(1, len(sample)//15)].to_string(index=False))

    # --- Trailing hit-rate validation (standalone, Path 2 pilot) — same idea as batter_power
    # above, reusing the SAME cached data (no new pull), just a different target event. ---
    print("\n" + "=" * 78)
    print("TRAILING HIT-RATE FEATURE — standalone validation (Path 2 pilot: hits prop)")
    print("=" * 78)
    hit_result = add_batter_hit_rate(batted)
    hit_coverage = hit_result["batter_hit_rate"].notna().mean()
    print(f"{hit_coverage*100:.0f}% of rows have a usable batter_hit_rate value "
          f"(rest are early-season / not enough prior history yet — expected, not a bug).")

    valid_rates = hit_result["batter_hit_rate"].dropna()
    if len(valid_rates):
        print(f"Trailing hit-rate distribution: min={valid_rates.min():.3f} "
              f"median={valid_rates.median():.3f} max={valid_rates.max():.3f} "
              f"(real MLB BABIP is roughly 0.29-0.30 — sanity check the median against that; "
              f"if it's way off, HIT_EVENTS or the window/threshold constants likely need a look)")

    print("\nSame everyday player, trailing hit rate over time:")
    hit_sample = hit_result[hit_result["batter"] == busiest_batter][
        ["game_date", "batter_hit_rate", "batter_hit_rate_n"]
    ].drop_duplicates("game_date").sort_values("game_date")
    print(hit_sample.iloc[::max(1, len(hit_sample)//15)].to_string(index=False))

    # --- Pitch-type matchup validation (standalone, doesn't touch calibrate_model.py yet) ---
    print("\n" + "=" * 78)
    print("PITCH-TYPE MATCHUP FEATURE — standalone validation")
    print("=" * 78)
    start_dt = result["game_date"].min().date().isoformat()
    end_dt = result["game_date"].max().date().isoformat()
    pitches = pull_full_pitch_data(start_dt, end_dt)

    matched = add_pitch_type_matchup(pitches, result)
    b_coverage = matched["batter_pitch_rv"].notna().mean()
    p_coverage = matched["pitcher_pitch_rv_allowed"].notna().mean()
    print(f"\n{b_coverage*100:.0f}% of batted balls have a usable batter_pitch_rv value")
    print(f"{p_coverage*100:.0f}% of batted balls have a usable pitcher_pitch_rv_allowed value")

    print("\nSample: same everyday player, their trailing run value against the specific pitch")
    print("type they hit, game by game (should vary — different pitch types, different values):")
    sample2 = matched[matched["batter"] == busiest_batter][
        ["game_date", "pitch_type", "batter_pitch_rv"]
    ].dropna(subset=["batter_pitch_rv"]).drop_duplicates(["game_date", "pitch_type"])
    print(sample2.sort_values("game_date").tail(15).to_string(index=False))  # ~15 evenly-spaced rows across the season
