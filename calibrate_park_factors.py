"""
calibrate_park_factors.py — Phase 2: real empirical park factors, computed
from actual historical HR outcomes instead of the hand-typed approximations
in hr_scanner.py's PARK_FACTORS table.

WHAT IT DOES
  1. Loads the batted-ball parquet(s) + weather_cache.csv from historical_data.py
  2. Joins them on game_pk to attach a real venue name to every batted ball
  3. Computes league_avg_hr_rate = HR / total batted balls, league-wide
  4. Computes the same rate per (venue, batter handedness)
  5. park_factor = park_hand_hr_rate / league_avg_hr_rate - 1
     (this is directly comparable to hr_scanner.py's PARK_FACTORS values —
     same +/- percentage format, just measured instead of guessed)

SAMPLE SIZE HONESTY: splitting by park AND handedness cuts your effective
sample per bucket fast. A bucket under MIN_SAMPLE gets flagged LOW CONF
instead of silently presented as equally trustworthy — a 14-day pilot will
throw a lot of these. That's expected; it's the signal to pull more data,
not a bug in this script.

RUN:
    python3 calibrate_park_factors.py
    python3 calibrate_park_factors.py data/batted_balls_2025-04-01_2025-09-30.parquet
"""

import glob
import sys

import pandas as pd

MIN_SAMPLE = 300  # batted balls needed in a (park, hand) bucket before we trust the number
SHRINKAGE_K = 300  # same technique as load_power_scores in hr_scanner_auto.py — blends in
                    # K league-average batted balls so a small sample (or a fluky 0 HRs)
                    # can't produce an impossible -100%/+infinite% park factor
RELOCATION_WARN_THRESHOLD = 15  # a non-primary venue with more games than this isn't a
                                 # neutral-site series (London, Mexico City, etc.) — it's a
                                 # real mid-season move and needs its own park factor, not exclusion


def get_primary_venues(batted: pd.DataFrame, weather: pd.DataFrame) -> tuple[set, pd.DataFrame]:
    """Every team's highest-game-count venue is treated as their real home park.
    Everything else (neutral-site series, one-off relocations) gets excluded —
    UNLESS it has enough games to look like a genuine mid-season move, in which
    case we flag it loudly instead of silently dropping real data."""
    home_venue = batted[["home_team", "game_pk"]].drop_duplicates().merge(
        weather[["game_pk", "venue"]], on="game_pk", how="left"
    )
    counts = home_venue.groupby(["home_team", "venue"])["game_pk"].nunique().reset_index(name="games")
    primary = counts.loc[counts.groupby("home_team")["games"].idxmax()]
    primary_pairs = set(zip(primary["home_team"], primary["venue"]))

    is_primary = counts.apply(lambda r: (r["home_team"], r["venue"]) in primary_pairs, axis=1)
    excluded = counts[~is_primary]
    real_relocations = excluded[excluded["games"] > RELOCATION_WARN_THRESHOLD]
    minor = excluded[excluded["games"] <= RELOCATION_WARN_THRESHOLD]

    if not minor.empty:
        print("\nExcluding minor/neutral-site venues (too few games to be a real home park):")
        for _, row in minor.iterrows():
            print(f"  {row['home_team']}: {row['venue']} ({row['games']} games)")

    if not real_relocations.empty:
        print("\n*** WARNING: these venues have ENOUGH games to be a real mid-season relocation, ***")
        print("*** NOT excluded — but they'll currently be lumped in under a different venue ***")
        print("*** name than the team's other home games unless you handle this manually:      ***")
        for _, row in real_relocations.iterrows():
            print(f"  {row['home_team']}: {row['venue']} ({row['games']} games)")

    return set(primary["venue"]), minor


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


def load_weather() -> pd.DataFrame:
    try:
        return pd.read_csv("data/weather_cache.csv")
    except FileNotFoundError:
        print("No weather_cache.csv found — run historical_data.py first.")
        sys.exit(1)


def compute_park_factors_home_road(batted: pd.DataFrame, primary_venues: set, weather: pd.DataFrame) -> pd.DataFrame:
    """Second method, for comparison: each team is its own control (home rate
    vs that same team's road rate) instead of pooling across different teams'
    rosters at a venue. Cancels out team-quality confounds (e.g. a team just
    having a bad pitching staff this year) that the naive pooled method can't
    tell apart from a real park effect. Caveat: this assumes a team's road
    schedule is a reasonably balanced sample of the league — true across a
    full MLB season's worth of opponents, less true in a small/lopsided
    sample, so treat divergences between this and the naive method as a
    signal to look closer, not an automatic 'this one's right.'"""
    venue_map = batted[["home_team", "game_pk"]].drop_duplicates().merge(
        weather[["game_pk", "venue"]], on="game_pk", how="left"
    )
    team_to_venue = venue_map[venue_map["venue"].isin(primary_venues)].groupby("home_team")["venue"].first()

    df = batted.copy()
    df["is_hr"] = (df["events"] == "home_run").astype(int)

    rows = []
    for team, venue in team_to_venue.items():
        for hand in ["L", "R"]:
            home = df[(df["home_team"] == team) & (df["stand"] == hand)]
            road = df[(df["away_team"] == team) & (df["stand"] == hand)]
            if len(home) < 50 or len(road) < 50:
                continue
            home_rate, road_rate = home["is_hr"].mean(), road["is_hr"].mean()
            rows.append({
                "venue": venue, "team": team, "stand": hand,
                "home_n": len(home), "road_n": len(road),
                "home_road_factor": home_rate / road_rate - 1 if road_rate > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def compute_park_factors(batted: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    primary_venues, _ = get_primary_venues(batted, weather)

    df = batted.merge(weather[["game_pk", "venue"]], on="game_pk", how="left")
    df["is_hr"] = (df["events"] == "home_run").astype(int)
    df = df.dropna(subset=["venue", "stand"])

    before = len(df)
    df = df[df["venue"].isin(primary_venues)]
    print(f"\nDropped {before - len(df)} batted balls from non-primary venues, {len(df)} remain.")

    league_avg = df["is_hr"].mean()
    print(f"\nLeague-wide HR rate per batted ball: {league_avg:.4f}  (n={len(df)})")

    grouped = (
        df.groupby(["venue", "stand"])
        .agg(batted_balls=("is_hr", "count"), hr_rate=("is_hr", "mean"))
        .reset_index()
    )
    grouped["park_factor_raw"] = grouped["hr_rate"] / league_avg - 1
    grouped["shrunk_rate"] = (
        (grouped["batted_balls"] * grouped["hr_rate"] + SHRINKAGE_K * league_avg)
        / (grouped["batted_balls"] + SHRINKAGE_K)
    )
    grouped["park_factor"] = grouped["shrunk_rate"] / league_avg - 1  # use this one, not park_factor_raw
    grouped["confidence"] = grouped["batted_balls"].apply(
        lambda n: "OK" if n >= MIN_SAMPLE else f"LOW CONF (n={n}, want {MIN_SAMPLE}+)"
    )
    return grouped.sort_values(["venue", "stand"])


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    batted = load_batted_balls(path_arg)
    weather = load_weather()

    result = compute_park_factors(batted, weather)

    print("\n" + "=" * 78)
    print("EMPIRICAL PARK FACTORS (compare against hr_scanner.py's hardcoded table)")
    print("=" * 78)
    for _, row in result.iterrows():
        print(f"  {row['venue']:<28} {row['stand']}  park_factor={row['park_factor']:+.3f}  "
              f"(raw {row['park_factor_raw']:+.3f})  hr_rate={row['hr_rate']:.4f}  {row['confidence']}")

    out_path = "data/park_factors_empirical.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved full table to {out_path}")

    low_conf = (result["confidence"] != "OK").sum()
    if low_conf:
        print(f"\n{low_conf} of {len(result)} (park, hand) buckets are LOW CONF on this data. "
              f"That's expected on a pilot-sized pull — this is the signal to scale up "
              f"historical_data.py to a full season before trusting these for real.")

    # --- comparison method: home/road, team-as-own-control ---
    primary_venues, _ = get_primary_venues(batted, weather)
    hr_result = compute_park_factors_home_road(batted, primary_venues, weather)

    print("\n" + "=" * 88)
    print("COMPARISON: naive pooled method vs home/road method")
    print("=" * 88)
    merged = result.merge(hr_result, on=["venue", "stand"], how="inner")
    merged["divergence"] = (merged["park_factor"] - merged["home_road_factor"]).abs()
    merged["park_factor_blended"] = (merged["park_factor"] + merged["home_road_factor"]) / 2
    merged["agreement"] = merged["divergence"].apply(lambda d: "DIVERGENT — unresolved, don't trust yet" if d > 0.15 else "converged")
    merged = merged.sort_values("divergence", ascending=False)
    for _, row in merged.iterrows():
        print(f"  {row['venue']:<28} {row['stand']}  pooled={row['park_factor']:+.3f}  "
              f"home/road={row['home_road_factor']:+.3f}  blended={row['park_factor_blended']:+.3f}  {row['agreement']}")

    blend_path = "data/park_factors_blended.csv"
    merged.to_csv(blend_path, index=False)
    print(f"\nSaved blended table to {blend_path}")
    n_divergent = (merged["agreement"] != "converged").sum()
    print(f"{len(merged)-n_divergent} converged, {n_divergent} still divergent and flagged unresolved.")
