"""
calibrate_hits_model.py — Path 2 pilot: the hits-prop version of
calibrate_model.py. Same regression methodology (logistic regression per
batter handedness, controlling for park, weather, and road-only team-quality
proxies simultaneously), just targeting is_hit instead of is_hr and
batter_hit_rate instead of batter_power.

WHY A SEPARATE FILE, NOT A PARAMETERIZED calibrate_model.py: every function
in calibrate_model.py that does anything with the fitted model (fit_model,
extract_park_factors, extract_weather_factors, extract_batter_power_factor,
extract_matchup_factor) has the target/feature column names baked directly
into formula strings and prediction-input DataFrames — genuinely
parameterizable, but calibrate_model.py runs in production every single day
via daily_update.py. The safer trade-off is some duplicated structure here
in exchange for zero risk to the live HR pipeline; the same call this
project already made for historical_features.py's add_batter_power (left
untouched rather than refactored onto the shared _trailing_by_group engine).

WHAT'S REUSED, NOT DUPLICATED: load_batted_balls, load_weather,
get_primary_venues, categorize_wind, bucket_temp, bucket_speed are all
genuinely prop-agnostic (no is_hr/batter_power anywhere in them) — imported
directly from calibrate_model.py rather than copy-pasted, so a fix to one
of those automatically applies here too.

MODEL: logistic regression, fit separately for L and R batters:
    is_hit ~ C(venue) + offense_quality + pitching_quality + batter_hit_rate
             + batter_pitch_rv + pitcher_pitch_rv_allowed
             + C(temp_bucket) + C(wind_dir) + C(speed_bucket)
Same significance rule, same reference-baseline-averaged-across-all-venues
approach (not one arbitrary reference park) that calibrate_model.py already
fixed after finding Fenway-as-reference was quietly inflating every other
park's number.

SETUP: pip install statsmodels
RUN:
    python3 calibrate_hits_model.py
    python3 calibrate_hits_model.py data/batted_balls_2025-04-01_2025-09-30.parquet
"""

import sys

import pandas as pd
import statsmodels.formula.api as smf

from historical_features import add_batter_hit_rate, pull_full_pitch_data, add_pitch_type_matchup
from calibrate_model import (
    load_batted_balls, load_weather, get_primary_venues,
    categorize_wind, bucket_temp, bucket_speed,
    MIN_TEAM_ROAD_SAMPLE,
)

HIT_EVENTS = {"single", "double", "triple", "home_run"}  # kept in sync with historical_features.HIT_EVENTS


def build_hits_dataset(batted: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Hits-prop version of calibrate_model.build_dataset. Identical
    structure — only the target event and the trailing-form feature differ."""
    primary_venues = get_primary_venues(batted, weather)

    batted = add_batter_hit_rate(batted)

    start_dt = batted["game_date"].min().date().isoformat()
    end_dt = batted["game_date"].max().date().isoformat()
    pitches = pull_full_pitch_data(start_dt, end_dt)
    batted = add_pitch_type_matchup(pitches, batted)

    df = batted.merge(weather, on="game_pk", how="left")
    df["is_hit"] = df["events"].isin(HIT_EVENTS).astype(int)
    df = df.dropna(subset=["venue", "stand", "home_team", "away_team", "inning_topbot"])
    df = df[df["venue"].isin(primary_venues)]

    before = len(df)
    df = df.dropna(subset=["batter_hit_rate", "batter_pitch_rv", "pitcher_pitch_rv_allowed"])
    print(f"Dropped {before - len(df)} rows missing batter_hit_rate, batter_pitch_rv, or "
          f"pitcher_pitch_rv_allowed (early season / too little history), {len(df)} remain.")

    df["batting_team"] = df["away_team"].where(df["inning_topbot"] == "Top", df["home_team"])
    df["pitching_team"] = df["home_team"].where(df["inning_topbot"] == "Top", df["away_team"])

    # road-only quality proxies, computed on is_hit -- same circularity-avoidance reasoning
    # as calibrate_model.py, just measuring "does this team hit for average" instead of power
    road_batting = df[df["batting_team"] == df["away_team"]]
    offense_quality = road_batting.groupby("batting_team")["is_hit"].agg(["mean", "count"])
    offense_quality = offense_quality[offense_quality["count"] >= MIN_TEAM_ROAD_SAMPLE]["mean"]

    road_pitching = df[df["pitching_team"] == df["away_team"]]
    pitching_quality = road_pitching.groupby("pitching_team")["is_hit"].agg(["mean", "count"])
    pitching_quality = pitching_quality[pitching_quality["count"] >= MIN_TEAM_ROAD_SAMPLE]["mean"]

    df["offense_quality"] = df["batting_team"].map(offense_quality)
    df["pitching_quality"] = df["pitching_team"].map(pitching_quality)
    before = len(df)
    df = df.dropna(subset=["offense_quality", "pitching_quality"])
    print(f"Dropped {before - len(df)} rows lacking a reliable team-quality proxy, {len(df)} remain.")

    df["condition_lower"] = df["condition"].astype(str).str.lower()
    df[["wind_speed", "wind_dir"]] = df["wind"].apply(lambda w: pd.Series(categorize_wind(w)))
    df["temp_bucket"] = df["temp"].apply(bucket_temp)
    df["speed_bucket"] = df["wind_speed"].apply(bucket_speed)

    return df


def fit_hits_model(df: pd.DataFrame, hand: str):
    sub = df[df["stand"] == hand].copy()
    print(f"\nFitting {hand}-handed batter HITS model on {len(sub)} batted balls...")

    formula = ("is_hit ~ C(venue) + offense_quality + pitching_quality + batter_hit_rate "
               "+ batter_pitch_rv + pitcher_pitch_rv_allowed "
               "+ C(temp_bucket) + C(wind_dir) + C(speed_bucket)")
    model = smf.logit(formula, data=sub).fit(disp=0)

    avg_offense = sub["offense_quality"].mean()
    avg_pitching = sub["pitching_quality"].mean()
    avg_hit_rate = sub["batter_hit_rate"].mean()
    avg_batter_rv = sub["batter_pitch_rv"].mean()
    avg_pitcher_rv = sub["pitcher_pitch_rv_allowed"].mean()
    ref_temp = sub["temp_bucket"].mode()[0]
    ref_wind = sub["wind_dir"].mode()[0]
    ref_speed = sub["speed_bucket"].mode()[0]
    return model, sub, avg_offense, avg_pitching, avg_hit_rate, avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed


def extract_hits_park_factors(model, sub, hand, avg_offense, avg_pitching, avg_hit_rate,
                               avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed) -> pd.DataFrame:
    venues = sub["venue"].unique()
    base_covariates = pd.DataFrame({
        "offense_quality": [avg_offense], "pitching_quality": [avg_pitching], "batter_hit_rate": [avg_hit_rate],
        "batter_pitch_rv": [avg_batter_rv], "pitcher_pitch_rv_allowed": [avg_pitcher_rv],
        "temp_bucket": [ref_temp], "wind_dir": [ref_wind], "speed_bucket": [ref_speed],
    })

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


def extract_hits_weather_factors(model, sub, hand, avg_offense, avg_pitching, avg_hit_rate,
                                  avg_batter_rv, avg_pitcher_rv) -> pd.DataFrame:
    venues = sub["venue"].unique()
    combos = sub[["temp_bucket", "wind_dir", "speed_bucket"]].drop_duplicates()

    def avg_pred(temp, wind, speed):
        preds = []
        for v in venues:
            inp = pd.DataFrame({
                "venue": [v], "offense_quality": [avg_offense], "pitching_quality": [avg_pitching],
                "batter_hit_rate": [avg_hit_rate], "batter_pitch_rv": [avg_batter_rv],
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

    venue_avg_rate = {v: model.predict(pd.DataFrame({
        "venue": [v], "offense_quality": [avg_offense], "pitching_quality": [avg_pitching],
        "batter_hit_rate": [avg_hit_rate], "batter_pitch_rv": [avg_batter_rv],
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
            "batter_hit_rate": [avg_hit_rate], "batter_pitch_rv": [avg_batter_rv],
            "pitcher_pitch_rv_allowed": [avg_pitcher_rv],
            "temp_bucket": [temp], "wind_dir": [wind], "speed_bucket": [speed],
        })
        pred_summary = model.get_prediction(pred_input).summary_frame(alpha=0.05)
        ci_low, ci_high = pred_summary["ci_lower"].iloc[0], pred_summary["ci_upper"].iloc[0]
        ref_pred = pred_summary["predicted"].iloc[0]
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


def extract_batter_hitrate_factor(model, sub, hand, avg_offense, avg_pitching, avg_batter_rv,
                                   avg_pitcher_rv, ref_temp, ref_wind, ref_speed) -> pd.DataFrame:
    venues = sub["venue"].unique()
    p10, p50, p90 = sub["batter_hit_rate"].quantile([0.10, 0.50, 0.90])

    def avg_pred(rate_value):
        preds = []
        for v in venues:
            inp = pd.DataFrame({
                "venue": [v], "offense_quality": [avg_offense], "pitching_quality": [avg_pitching],
                "batter_hit_rate": [rate_value], "batter_pitch_rv": [avg_batter_rv],
                "pitcher_pitch_rv_allowed": [avg_pitcher_rv],
                "temp_bucket": [ref_temp], "wind_dir": [ref_wind], "speed_bucket": [ref_speed],
            })
            preds.append(model.predict(inp).iloc[0])
        return sum(preds) / len(preds)

    pred_low, pred_mid, pred_high = avg_pred(p10), avg_pred(p50), avg_pred(p90)

    ref_venue = venues[0]
    low_input = pd.DataFrame({"venue": [ref_venue], "offense_quality": [avg_offense],
                               "pitching_quality": [avg_pitching], "batter_hit_rate": [p10],
                               "batter_pitch_rv": [avg_batter_rv], "pitcher_pitch_rv_allowed": [avg_pitcher_rv],
                               "temp_bucket": [ref_temp], "wind_dir": [ref_wind], "speed_bucket": [ref_speed]})
    high_input = low_input.copy()
    high_input["batter_hit_rate"] = p90
    low_summary = model.get_prediction(low_input).summary_frame(alpha=0.05)
    high_summary = model.get_prediction(high_input).summary_frame(alpha=0.05)

    return pd.DataFrame([{
        "factor_type": "batter_hit_rate", "key": f"p10={p10:.3f} vs p90={p90:.3f} (median={p50:.3f})", "stand": hand,
        "pred_at_p10": pred_low, "pred_at_p50": pred_mid, "pred_at_p90": pred_high,
        "factor": pred_high / pred_low - 1,
        "factor_ci_low": high_summary["ci_lower"].iloc[0] / low_summary["ci_upper"].iloc[0] - 1,
        "factor_ci_high": high_summary["ci_upper"].iloc[0] / max(low_summary["ci_lower"].iloc[0], 1e-6) - 1,
        "significant": "yes" if low_summary["ci_upper"].iloc[0] < high_summary["ci_lower"].iloc[0] else "no (CIs overlap)",
    }])


def extract_hits_matchup_factor(model, sub, hand, avg_offense, avg_pitching, avg_hit_rate,
                                 ref_temp, ref_wind, ref_speed) -> pd.DataFrame:
    venues = sub["venue"].unique()
    b_p10, b_p50, b_p90 = sub["batter_pitch_rv"].quantile([0.10, 0.50, 0.90])
    p_p10, p_p50, p_p90 = sub["pitcher_pitch_rv_allowed"].quantile([0.10, 0.50, 0.90])

    def avg_pred(batter_rv, pitcher_rv):
        preds = []
        for v in venues:
            inp = pd.DataFrame({
                "venue": [v], "offense_quality": [avg_offense], "pitching_quality": [avg_pitching],
                "batter_hit_rate": [avg_hit_rate], "batter_pitch_rv": [batter_rv],
                "pitcher_pitch_rv_allowed": [pitcher_rv],
                "temp_bucket": [ref_temp], "wind_dir": [ref_wind], "speed_bucket": [ref_speed],
            })
            preds.append(model.predict(inp).iloc[0])
        return sum(preds) / len(preds)

    pred_bad = avg_pred(b_p10, p_p10)
    pred_mid = avg_pred(b_p50, p_p50)
    pred_good = avg_pred(b_p90, p_p90)

    ref_venue = venues[0]
    bad_input = pd.DataFrame({"venue": [ref_venue], "offense_quality": [avg_offense],
                               "pitching_quality": [avg_pitching], "batter_hit_rate": [avg_hit_rate],
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

    df = build_hits_dataset(batted, weather)

    park_results, weather_results, hitrate_results, matchup_results = [], [], [], []
    for hand in ["L", "R"]:
        (model, sub, avg_offense, avg_pitching, avg_hit_rate, avg_batter_rv,
         avg_pitcher_rv, ref_temp, ref_wind, ref_speed) = fit_hits_model(df, hand)
        park_results.append(extract_hits_park_factors(model, sub, hand, avg_offense, avg_pitching, avg_hit_rate, avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed))
        weather_results.append(extract_hits_weather_factors(model, sub, hand, avg_offense, avg_pitching, avg_hit_rate, avg_batter_rv, avg_pitcher_rv))
        hitrate_results.append(extract_batter_hitrate_factor(model, sub, hand, avg_offense, avg_pitching, avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed))
        matchup_results.append(extract_hits_matchup_factor(model, sub, hand, avg_offense, avg_pitching, avg_hit_rate, ref_temp, ref_wind, ref_speed))

    park_df = pd.concat(park_results, ignore_index=True)
    weather_df = pd.concat(weather_results, ignore_index=True)
    hitrate_df = pd.concat(hitrate_results, ignore_index=True)
    matchup_df = pd.concat(matchup_results, ignore_index=True)

    print("\n" + "=" * 88)
    print("MODEL-BASED PARK FACTORS (HITS)")
    print("=" * 88)
    for _, row in park_df.sort_values(["key", "stand"]).iterrows():
        print(f"  {row['key']:<28} {row['stand']}  factor={row['factor']:+.3f}  "
              f"CI=[{row['factor_ci_low']:+.3f}, {row['factor_ci_high']:+.3f}]  significant={row['significant']}")

    print("\n" + "=" * 88)
    print("MODEL-BASED WEATHER FACTORS (HITS) (park, team quality, batter hit rate, matchup controlled for)")
    print("=" * 88)
    for _, row in weather_df.sort_values(["stand", "factor"], ascending=[True, False]).iterrows():
        print(f"  {row['key']:<38} {row['stand']}  n={row['n']:<5}  factor={row['factor']:+.3f}  "
              f"CI=[{row['factor_ci_low']:+.3f}, {row['factor_ci_high']:+.3f}]  significant={row['significant']}")

    print("\n" + "=" * 88)
    print("MODEL-BASED BATTER HIT-RATE EFFECT (park, weather, team quality, matchup controlled for)")
    print("=" * 88)
    for _, row in hitrate_df.iterrows():
        print(f"  stand={row['stand']}  {row['key']}")
        print(f"    predicted hit rate: cold={row['pred_at_p10']:.4f}  median={row['pred_at_p50']:.4f}  hot={row['pred_at_p90']:.4f}")
        print(f"    swing (hot vs cold): {row['factor']:+.3f}  CI=[{row['factor_ci_low']:+.3f}, {row['factor_ci_high']:+.3f}]  significant={row['significant']}")

    print("\n" + "=" * 88)
    print("MODEL-BASED PITCH-TYPE MATCHUP EFFECT ON HITS (park, weather, team quality, hit rate controlled for)")
    print("=" * 88)
    for _, row in matchup_df.iterrows():
        print(f"  stand={row['stand']}  {row['key']}")
        print(f"    predicted hit rate: bad matchup={row['pred_bad']:.4f}  typical={row['pred_mid']:.4f}  good matchup={row['pred_good']:.4f}")
        print(f"    swing (good vs bad): {row['factor']:+.3f}  CI=[{row['factor_ci_low']:+.3f}, {row['factor_ci_high']:+.3f}]  significant={row['significant']}")

    park_df.to_csv("data/hits_park_factors_model.csv", index=False)
    weather_df.to_csv("data/hits_weather_factors_model.csv", index=False)
    hitrate_df.to_csv("data/hits_batter_hitrate_factor_model.csv", index=False)
    matchup_df.to_csv("data/hits_matchup_factor_model.csv", index=False)
    print(f"\nSaved to data/hits_park_factors_model.csv, data/hits_weather_factors_model.csv, "
          f"data/hits_batter_hitrate_factor_model.csv, data/hits_matchup_factor_model.csv")

    for label, results in [("park/hand", park_df), ("weather/hand", weather_df)]:
        n_sig = (results["significant"] == "yes").sum()
        print(f"{n_sig} of {len(results)} {label} effects are statistically significant even after "
              f"controlling for the other factors.")
