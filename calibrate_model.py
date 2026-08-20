"""
calibrate_model.py — Phase 4a: a real regression instead of sequential ratio
tricks. Controls for park, weather, batting-team quality, and pitching-team
quality all at once, so a park's coefficient reflects the park itself, not
"whichever teams happened to play there this season."

WHY THIS EXISTS: the naive pooled method (Phase 2) confounds park with team
quality. The home/road method we tried as a fix trades that confound for a
different one — it's the classic Bill James formula, and it's sensitive to
whether a team's road schedule is a balanced sample of the league (proven
with a synthetic test: an imbalanced 3-4 team toy schedule badly distorted
it). A regression can hold multiple confounds constant simultaneously
instead of hoping one ratio cancels one problem.

KEY DESIGN CHOICE — road-only team-quality proxies: each team's offensive
and pitching quality is measured ONLY from their own road games (using
inning_topbot to correctly identify who's actually batting on a given pitch,
not just home/away for the game). This keeps the quality proxy from ever
being contaminated by the team's own home park — exactly the circularity
that would defeat the point.

MODEL: logistic regression, fit separately for L and R batters:
    is_hr ~ C(venue) + offense_quality + pitching_quality
            + C(temp_bucket) + C(wind_dir) + C(speed_bucket)
Park factors are read off as: predicted P(HR) at that park (holding quality
and weather at league-average) vs. predicted P(HR) at a reference park,
expressed as a % difference — same format as Phases 2 and 3, just properly
isolated this time.

SETUP: pip install statsmodels
RUN:
    python3 calibrate_model.py
    python3 calibrate_model.py data/batted_balls_2025-04-01_2025-09-30.parquet
"""

import glob
import sys

import pandas as pd
import statsmodels.formula.api as smf

from historical_features import add_batter_power, pull_full_pitch_data, add_pitch_type_matchup

DOME_CONDITIONS = {"roof closed", "dome"}
RELOCATION_WARN_THRESHOLD = 15
MIN_TEAM_ROAD_SAMPLE = 200  # need at least this many road batted balls to trust a team's quality proxy


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


def get_primary_venues(batted: pd.DataFrame, weather: pd.DataFrame) -> set:
    home_venue = batted[["home_team", "game_pk"]].drop_duplicates().merge(
        weather[["game_pk", "venue"]], on="game_pk", how="left"
    )
    counts = home_venue.groupby(["home_team", "venue"])["game_pk"].nunique().reset_index(name="games")
    primary = counts.loc[counts.groupby("home_team")["games"].idxmax()]
    return set(primary["venue"])


def categorize_wind(wind_str) -> tuple:
    s = str(wind_str).strip().lower()
    if not s or s == "nan":
        return 0.0, "Calm"
    try:
        speed = float(s.split("mph")[0].strip())
    except ValueError:
        speed = 0.0
    if "out" in s:
        return speed, "Out"
    if "in from" in s:
        return speed, "In"
    if "to r" in s or "to l" in s:
        return speed, "Cross"
    if "calm" in s or "none" in s:
        return speed, "Calm"
    if "varies" in s:
        return speed, "Varies"
    return speed, "Unknown"


def bucket_temp(temp) -> str:
    try:
        t = float(temp)
    except (ValueError, TypeError):
        return "Unknown"
    if t < 60: return "<60"
    if t < 70: return "60-70"
    if t < 80: return "70-80"
    if t < 90: return "80-90"
    return "90+"


def bucket_speed(speed: float) -> str:
    if speed < 5: return "0-5"
    if speed < 10: return "5-10"
    if speed < 15: return "10-15"
    return "15+"


def build_dataset(batted: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    primary_venues = get_primary_venues(batted, weather)

    # batter_power computed on the FULL unfiltered batted-ball history first — a batter's
    # own recent form isn't tied to which park we end up modeling, so this needs every game
    # they played, not just games at parks that survive the primary-venue filter below.
    batted = add_batter_power(batted)

    # pitch-type matchup — needs the FULL pitch-level pull (not just contact), same date
    # range as the batted-ball data, so it can compute trailing batter-vs-pitch-type and
    # pitcher-vs-pitch-type signals the same way historical_features.py validated standalone.
    start_dt = batted["game_date"].min().date().isoformat()
    end_dt = batted["game_date"].max().date().isoformat()
    pitches = pull_full_pitch_data(start_dt, end_dt)
    batted = add_pitch_type_matchup(pitches, batted)

    df = batted.merge(weather, on="game_pk", how="left")
    df["is_hr"] = (df["events"] == "home_run").astype(int)
    df = df.dropna(subset=["venue", "stand", "home_team", "away_team", "inning_topbot"])
    df = df[df["venue"].isin(primary_venues)]

    before = len(df)
    df = df.dropna(subset=["batter_power", "batter_pitch_rv", "pitcher_pitch_rv_allowed"])
    print(f"Dropped {before - len(df)} rows missing batter_power, batter_pitch_rv, or "
          f"pitcher_pitch_rv_allowed (early season / too little history), {len(df)} remain.")

    # who's actually batting/pitching on THIS pitch, not just home/away for the game
    df["batting_team"] = df["away_team"].where(df["inning_topbot"] == "Top", df["home_team"])
    df["pitching_team"] = df["home_team"].where(df["inning_topbot"] == "Top", df["away_team"])

    # road-only quality proxies — never contaminated by the team's own park
    road_batting = df[df["batting_team"] == df["away_team"]]
    offense_quality = road_batting.groupby("batting_team")["is_hr"].agg(["mean", "count"])
    offense_quality = offense_quality[offense_quality["count"] >= MIN_TEAM_ROAD_SAMPLE]["mean"]

    road_pitching = df[df["pitching_team"] == df["away_team"]]
    pitching_quality = road_pitching.groupby("pitching_team")["is_hr"].agg(["mean", "count"])
    pitching_quality = pitching_quality[pitching_quality["count"] >= MIN_TEAM_ROAD_SAMPLE]["mean"]

    df["offense_quality"] = df["batting_team"].map(offense_quality)
    df["pitching_quality"] = df["pitching_team"].map(pitching_quality)
    before = len(df)
    df = df.dropna(subset=["offense_quality", "pitching_quality"])
    print(f"Dropped {before - len(df)} rows lacking a reliable team-quality proxy, {len(df)} remain.")

    # weather — dome games keep their rows (previously dropped entirely, which silently
    # gutted the sample for any team whose roof is closed most of the time, like loanDepot
    # and Globe Life Field, blowing up their confidence intervals). No special "Dome" label
    # needed: dome games already record "0 mph, None" for wind, which the existing parser
    # correctly reads as calm — a closed roof and a calm outdoor day look the same to a
    # batted ball, which is physically accurate, not an approximation.
    df["condition_lower"] = df["condition"].astype(str).str.lower()
    df[["wind_speed", "wind_dir"]] = df["wind"].apply(lambda w: pd.Series(categorize_wind(w)))
    df["temp_bucket"] = df["temp"].apply(bucket_temp)
    df["speed_bucket"] = df["wind_speed"].apply(bucket_speed)

    return df


def fit_model(df: pd.DataFrame, hand: str):
    """Fits the shared model once. Park, weather, batter-power, and pitch-type
    matchup factors all get read off the same fit — no need to refit for each."""
    sub = df[df["stand"] == hand].copy()
    print(f"\nFitting {hand}-handed batter model on {len(sub)} batted balls...")

    formula = ("is_hr ~ C(venue) + offense_quality + pitching_quality + batter_power "
               "+ batter_pitch_rv + pitcher_pitch_rv_allowed "
               "+ C(temp_bucket) + C(wind_dir) + C(speed_bucket)")
    model = smf.logit(formula, data=sub).fit(disp=0)

    avg_offense = sub["offense_quality"].mean()
    avg_pitching = sub["pitching_quality"].mean()
    avg_power = sub["batter_power"].mean()
    avg_batter_rv = sub["batter_pitch_rv"].mean()
    avg_pitcher_rv = sub["pitcher_pitch_rv_allowed"].mean()
    ref_temp = sub["temp_bucket"].mode()[0]
    ref_wind = sub["wind_dir"].mode()[0]
    ref_speed = sub["speed_bucket"].mode()[0]
    return model, sub, avg_offense, avg_pitching, avg_power, avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed


def extract_park_factors(model, sub, hand, avg_offense, avg_pitching, avg_power, avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed) -> pd.DataFrame:
    venues = sub["venue"].unique()
    base_covariates = pd.DataFrame({
        "offense_quality": [avg_offense], "pitching_quality": [avg_pitching], "batter_power": [avg_power],
        "batter_pitch_rv": [avg_batter_rv], "pitcher_pitch_rv_allowed": [avg_pitcher_rv],
        "temp_bucket": [ref_temp], "wind_dir": [ref_wind], "speed_bucket": [ref_speed],
    })

    # Baseline = average predicted probability ACROSS ALL PARKS, not one arbitrarily-chosen
    # real park. Comparing every venue to one specific other venue (whichever patsy/pandas
    # happens to order first) silently inflates every other park if that one reference park
    # happens to be genuinely below average — confirmed this was happening: the old version
    # used Fenway as the reference, which every prior phase of this project independently
    # found to be a real below-average park, so everything else was inflated by Fenway's own gap.
    all_venue_preds = []
    for v in venues:
        inp = base_covariates.copy()
        inp["venue"] = v
        all_venue_preds.append(model.predict(inp).iloc[0])
    baseline_pred = sum(all_venue_preds) / len(all_venue_preds)

    rows = []
    for venue in venues:
        pred_input = base_covariates.copy()
        pred_input["venue"] = venue
        # confidence interval on the predicted probability itself, not a per-coefficient
        # p-value — a p-value lookup structurally can't cover whichever venue patsy picks
        # as its own internal reference category, so every venue gets left out of that
        # check exactly once; a CI on the prediction has no such blind spot. Note: this CI
        # reflects uncertainty in THIS venue's own prediction only, not in the multi-park
        # average baseline itself — a simplification, flagged here rather than hidden.
        pred_summary = model.get_prediction(pred_input).summary_frame(alpha=0.05)
        pred = pred_summary["predicted"].iloc[0]
        ci_low, ci_high = pred_summary["ci_lower"].iloc[0], pred_summary["ci_upper"].iloc[0]
        significant = "yes" if (ci_low > baseline_pred or ci_high < baseline_pred) else "no (CI overlaps baseline)"
        rows.append({
            "factor_type": "park", "key": venue, "stand": hand,
            "factor": pred / baseline_pred - 1,
            "factor_ci_low": ci_low / baseline_pred - 1,
            "factor_ci_high": ci_high / baseline_pred - 1,
            "significant": significant,
        })
    return pd.DataFrame(rows).sort_values("factor", ascending=False)


def extract_weather_factors(model, sub, hand, avg_offense, avg_pitching, avg_power, avg_batter_rv, avg_pitcher_rv) -> pd.DataFrame:
    """Same idea as park factors, but marginalized the other way: for each
    real (temp, wind, speed) combination in the data, average the predicted
    probability across every park (not weighted by how often that park shows
    up), so no single venue's quirks drive the weather estimate the way
    Fenway accidentally drove every park estimate before that fix."""
    venues = sub["venue"].unique()
    combos = sub[["temp_bucket", "wind_dir", "speed_bucket"]].drop_duplicates()

    def avg_pred(temp, wind, speed):
        preds = []
        for v in venues:
            inp = pd.DataFrame({
                "venue": [v], "offense_quality": [avg_offense], "pitching_quality": [avg_pitching],
                "batter_power": [avg_power], "batter_pitch_rv": [avg_batter_rv],
                "pitcher_pitch_rv_allowed": [avg_pitcher_rv],
                "temp_bucket": [temp], "wind_dir": [wind], "speed_bucket": [speed],
            })
            preds.append(model.predict(inp).iloc[0])
        return sum(preds) / len(preds)

    combo_preds = {}
    for row in combos.itertuples(index=False):
        combo_preds[(row.temp_bucket, row.wind_dir, row.speed_bucket)] = avg_pred(
            row.temp_bucket, row.wind_dir, row.speed_bucket
        )
    baseline_pred = sum(combo_preds.values()) / len(combo_preds)

    # For confidence intervals: pick ONE representative venue (closest to the
    # overall average predicted rate across all venues) as a fixed anchor point.
    # Averaging a CI properly across venues needs the covariance between venue
    # and weather coefficients, which get_prediction() doesn't hand us directly —
    # this is the same kind of simplification already flagged in extract_park_factors,
    # just applied in the other direction. The point estimates above are the
    # properly-averaged ones; only the CI width is approximated.
    venue_avg_rate = {v: model.predict(pd.DataFrame({
        "venue": [v], "offense_quality": [avg_offense], "pitching_quality": [avg_pitching],
        "batter_power": [avg_power], "batter_pitch_rv": [avg_batter_rv],
        "pitcher_pitch_rv_allowed": [avg_pitcher_rv],
        "temp_bucket": [sub["temp_bucket"].mode()[0]], "wind_dir": [sub["wind_dir"].mode()[0]],
        "speed_bucket": [sub["speed_bucket"].mode()[0]],
    })).iloc[0] for v in venues}
    overall_avg = sum(venue_avg_rate.values()) / len(venue_avg_rate)
    ref_venue = min(venue_avg_rate, key=lambda v: abs(venue_avg_rate[v] - overall_avg))

    rows = []
    for (temp, wind, speed), pred in combo_preds.items():
        n = len(sub[(sub["temp_bucket"] == temp) & (sub["wind_dir"] == wind) & (sub["speed_bucket"] == speed)])
        pred_input = pd.DataFrame({
            "venue": [ref_venue], "offense_quality": [avg_offense], "pitching_quality": [avg_pitching],
            "batter_power": [avg_power], "batter_pitch_rv": [avg_batter_rv],
            "pitcher_pitch_rv_allowed": [avg_pitcher_rv],
            "temp_bucket": [temp], "wind_dir": [wind], "speed_bucket": [speed],
        })
        pred_summary = model.get_prediction(pred_input).summary_frame(alpha=0.05)
        ci_low, ci_high = pred_summary["ci_lower"].iloc[0], pred_summary["ci_upper"].iloc[0]
        ref_pred = pred_summary["predicted"].iloc[0]
        # rescale the reference-venue CI width onto the properly-averaged point estimate,
        # rather than reporting the reference venue's own (possibly different) point estimate
        half_width_ratio = (ci_high - ci_low) / (2 * ref_pred) if ref_pred > 0 else 0
        ci_low_adj = pred * (1 - half_width_ratio)
        ci_high_adj = pred * (1 + half_width_ratio)
        significant = "yes" if (ci_low_adj > baseline_pred or ci_high_adj < baseline_pred) else "no (CI overlaps baseline)"
        rows.append({
            "factor_type": "weather", "key": f"temp={temp} wind={wind} speed={speed}mph", "stand": hand,
            "factor": pred / baseline_pred - 1,
            "factor_ci_low": ci_low_adj / baseline_pred - 1,
            "factor_ci_high": ci_high_adj / baseline_pred - 1,
            "significant": significant,
            "n": n,
        })
    return pd.DataFrame(rows).sort_values("factor", ascending=False)


def extract_batter_power_factor(model, sub, hand, avg_offense, avg_pitching, avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed) -> pd.DataFrame:
    """batter_power is continuous, not categorical — there's no 'this park vs
    that park' comparison to make. Instead: what's the swing in predicted HR
    probability between a cold hitter and a hot one, holding park/weather/team
    quality at their averages? Uses the 10th and 90th percentile of REAL
    batter_power values in the data (not some arbitrary made-up spread), so
    'weak' and 'strong' mean what they actually mean for this dataset."""
    venues = sub["venue"].unique()
    p10, p50, p90 = sub["batter_power"].quantile([0.10, 0.50, 0.90])

    def avg_pred(power_value):
        preds = []
        for v in venues:
            inp = pd.DataFrame({
                "venue": [v], "offense_quality": [avg_offense], "pitching_quality": [avg_pitching],
                "batter_power": [power_value], "batter_pitch_rv": [avg_batter_rv],
                "pitcher_pitch_rv_allowed": [avg_pitcher_rv],
                "temp_bucket": [ref_temp], "wind_dir": [ref_wind], "speed_bucket": [ref_speed],
            })
            preds.append(model.predict(inp).iloc[0])
        return sum(preds) / len(preds)

    pred_low, pred_mid, pred_high = avg_pred(p10), avg_pred(p50), avg_pred(p90)

    # CI on the swing: use the same representative-venue simplification as weather,
    # applied to the batter_power coefficient specifically via get_prediction.
    ref_venue = venues[0]
    low_input = pd.DataFrame({"venue": [ref_venue], "offense_quality": [avg_offense],
                               "pitching_quality": [avg_pitching], "batter_power": [p10],
                               "batter_pitch_rv": [avg_batter_rv], "pitcher_pitch_rv_allowed": [avg_pitcher_rv],
                               "temp_bucket": [ref_temp], "wind_dir": [ref_wind], "speed_bucket": [ref_speed]})
    high_input = low_input.copy()
    high_input["batter_power"] = p90
    low_summary = model.get_prediction(low_input).summary_frame(alpha=0.05)
    high_summary = model.get_prediction(high_input).summary_frame(alpha=0.05)

    return pd.DataFrame([{
        "factor_type": "batter_power", "key": f"p10={p10:.3f} vs p90={p90:.3f} (median={p50:.3f})", "stand": hand,
        "pred_at_p10": pred_low, "pred_at_p50": pred_mid, "pred_at_p90": pred_high,
        "factor": pred_high / pred_low - 1,  # how much more likely a hot hitter is to homer vs a cold one
        # smallest plausible swing: hot batter's worst case over cold batter's best case
        "factor_ci_low": high_summary["ci_lower"].iloc[0] / low_summary["ci_upper"].iloc[0] - 1,
        # largest plausible swing: hot batter's best case over cold batter's worst case
        "factor_ci_high": high_summary["ci_upper"].iloc[0] / max(low_summary["ci_lower"].iloc[0], 1e-6) - 1,
        "significant": "yes" if low_summary["ci_upper"].iloc[0] < high_summary["ci_lower"].iloc[0] else "no (CIs overlap)",
    }])


def extract_matchup_factor(model, sub, hand, avg_offense, avg_pitching, avg_power, ref_temp, ref_wind, ref_speed) -> pd.DataFrame:
    """Same percentile-swing idea as batter_power, but for the pitch-type
    matchup — and this one is genuinely two-dimensional: a 'good matchup'
    means the batter's trailing performance against this pitch type is
    strong AND the pitcher's trailing performance allowing it is weak
    (both p90 in the batter's favor), a 'bad matchup' is both at p10.
    This is the direct answer to the original Ben Rice vs Chris Sale
    question — is a hot-batter/cold-pitcher pitch-type overlap actually
    worth anything once park, weather, and overall power are held constant."""
    venues = sub["venue"].unique()
    b_p10, b_p50, b_p90 = sub["batter_pitch_rv"].quantile([0.10, 0.50, 0.90])
    p_p10, p_p50, p_p90 = sub["pitcher_pitch_rv_allowed"].quantile([0.10, 0.50, 0.90])

    def avg_pred(batter_rv, pitcher_rv):
        preds = []
        for v in venues:
            inp = pd.DataFrame({
                "venue": [v], "offense_quality": [avg_offense], "pitching_quality": [avg_pitching],
                "batter_power": [avg_power], "batter_pitch_rv": [batter_rv],
                "pitcher_pitch_rv_allowed": [pitcher_rv],
                "temp_bucket": [ref_temp], "wind_dir": [ref_wind], "speed_bucket": [ref_speed],
            })
            preds.append(model.predict(inp).iloc[0])
        return sum(preds) / len(preds)

    # bad matchup: batter cold on this pitch (p10) AND pitcher dominant with it (p10, i.e.
    # suppresses runs) -> use pitcher's p10 (most negative = best for pitcher) as "tough"
    pred_bad = avg_pred(b_p10, p_p10)
    pred_mid = avg_pred(b_p50, p_p50)
    pred_good = avg_pred(b_p90, p_p90)

    ref_venue = venues[0]
    bad_input = pd.DataFrame({"venue": [ref_venue], "offense_quality": [avg_offense],
                               "pitching_quality": [avg_pitching], "batter_power": [avg_power],
                               "batter_pitch_rv": [b_p10], "pitcher_pitch_rv_allowed": [p_p10],
                               "temp_bucket": [ref_temp], "wind_dir": [ref_wind], "speed_bucket": [ref_speed]})
    good_input = bad_input.copy()
    good_input["batter_pitch_rv"] = b_p90
    good_input["pitcher_pitch_rv_allowed"] = p_p90
    bad_summary = model.get_prediction(bad_input).summary_frame(alpha=0.05)
    good_summary = model.get_prediction(good_input).summary_frame(alpha=0.05)

    return pd.DataFrame([{
        "factor_type": "matchup", "stand": hand,
        "key": f"bad(batter_rv={b_p10:.3f},pitcher_rv={p_p10:.3f}) vs good(batter_rv={b_p90:.3f},pitcher_rv={p_p90:.3f})",
        "pred_bad": pred_bad, "pred_mid": pred_mid, "pred_good": pred_good,
        "factor": pred_good / pred_bad - 1,
        "factor_ci_low": good_summary["ci_lower"].iloc[0] / bad_summary["ci_upper"].iloc[0] - 1,
        "factor_ci_high": good_summary["ci_upper"].iloc[0] / max(bad_summary["ci_lower"].iloc[0], 1e-6) - 1,
        "significant": "yes" if bad_summary["ci_upper"].iloc[0] < good_summary["ci_lower"].iloc[0] else "no (CIs overlap)",
    }])


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    batted = load_batted_balls(path_arg)
    weather = load_weather()

    df = build_dataset(batted, weather)

    park_results, weather_results, power_results, matchup_results = [], [], [], []
    for hand in ["L", "R"]:
        model, sub, avg_offense, avg_pitching, avg_power, avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed = fit_model(df, hand)
        park_results.append(extract_park_factors(model, sub, hand, avg_offense, avg_pitching, avg_power, avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed))
        weather_results.append(extract_weather_factors(model, sub, hand, avg_offense, avg_pitching, avg_power, avg_batter_rv, avg_pitcher_rv))
        power_results.append(extract_batter_power_factor(model, sub, hand, avg_offense, avg_pitching, avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed))
        matchup_results.append(extract_matchup_factor(model, sub, hand, avg_offense, avg_pitching, avg_power, ref_temp, ref_wind, ref_speed))

    park_df = pd.concat(park_results, ignore_index=True)
    weather_df = pd.concat(weather_results, ignore_index=True)
    power_df = pd.concat(power_results, ignore_index=True)
    matchup_df = pd.concat(matchup_results, ignore_index=True)

    print("\n" + "=" * 88)
    print("MODEL-BASED PARK FACTORS")
    print("=" * 88)
    for _, row in park_df.sort_values(["key", "stand"]).iterrows():
        print(f"  {row['key']:<28} {row['stand']}  factor={row['factor']:+.3f}  "
              f"CI=[{row['factor_ci_low']:+.3f}, {row['factor_ci_high']:+.3f}]  significant={row['significant']}")

    print("\n" + "=" * 88)
    print("MODEL-BASED WEATHER FACTORS (park, team quality, batter power, matchup controlled for)")
    print("=" * 88)
    for _, row in weather_df.sort_values(["stand", "factor"], ascending=[True, False]).iterrows():
        print(f"  {row['key']:<38} {row['stand']}  n={row['n']:<5}  factor={row['factor']:+.3f}  "
              f"CI=[{row['factor_ci_low']:+.3f}, {row['factor_ci_high']:+.3f}]  significant={row['significant']}")

    print("\n" + "=" * 88)
    print("MODEL-BASED BATTER POWER EFFECT (park, weather, team quality, matchup controlled for)")
    print("=" * 88)
    for _, row in power_df.iterrows():
        print(f"  stand={row['stand']}  {row['key']}")
        print(f"    predicted HR rate: cold={row['pred_at_p10']:.4f}  median={row['pred_at_p50']:.4f}  hot={row['pred_at_p90']:.4f}")
        print(f"    swing (hot vs cold): {row['factor']:+.3f}  CI=[{row['factor_ci_low']:+.3f}, {row['factor_ci_high']:+.3f}]  significant={row['significant']}")

    print("\n" + "=" * 88)
    print("MODEL-BASED PITCH-TYPE MATCHUP EFFECT (park, weather, team quality, batter power controlled for)")
    print("=" * 88)
    for _, row in matchup_df.iterrows():
        print(f"  stand={row['stand']}  {row['key']}")
        print(f"    predicted HR rate: bad matchup={row['pred_bad']:.4f}  typical={row['pred_mid']:.4f}  good matchup={row['pred_good']:.4f}")
        print(f"    swing (good vs bad): {row['factor']:+.3f}  CI=[{row['factor_ci_low']:+.3f}, {row['factor_ci_high']:+.3f}]  significant={row['significant']}")

    park_df.to_csv("data/park_factors_model.csv", index=False)
    weather_df.to_csv("data/weather_factors_model.csv", index=False)
    power_df.to_csv("data/batter_power_factor_model.csv", index=False)
    matchup_df.to_csv("data/matchup_factor_model.csv", index=False)
    print(f"\nSaved to data/park_factors_model.csv, data/weather_factors_model.csv, "
          f"data/batter_power_factor_model.csv, data/matchup_factor_model.csv")

    for label, results in [("park/hand", park_df), ("weather/hand", weather_df)]:
        n_sig = (results["significant"] == "yes").sum()
        print(f"{n_sig} of {len(results)} {label} effects are statistically significant even after "
              f"controlling for the other factors.")
