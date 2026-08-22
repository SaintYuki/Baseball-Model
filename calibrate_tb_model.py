"""
calibrate_tb_model.py — Path 2, prop 2: total bases. Same regression
methodology as calibrate_hits_model.py (logistic regression per handedness,
park + weather + team-quality + trailing form + matchup, controlled for
simultaneously) — but with one real structural difference the other two
props didn't need.

WHY THIS ONE IS DIFFERENT: "2+ total bases" is a whole-GAME fact, not a
per-batted-ball one — a batter reaches it via one extra-base hit, OR via
two singles in the same game. HR and hits both fit cleanly at the
per-batted-ball level (a batted ball either is or isn't a HR/hit). Total
bases can't: fitting the regression on individual batted-ball rows would
mean each of a multi-hit batter's rows gets treated as an independent
observation of the SAME game's outcome, which both overcounts the sample
and — worse — can't even express "two singles added up to the over."

WHAT THIS FILE DOES ABOUT IT: build_tb_dataset computes everything at the
usual per-batted-ball grain first (trailing TB rate, pitch-type matchup,
weather, team quality), then GROUPS BY (batter, game_pk) before fitting —
one row per batter-game, is_2plus_tb computed from the TRUE sum of total
bases across every batted ball that batter had that game (not filtered by
covariate trust first, so a thin-history pitch type on one batted ball
can't silently undercount a real multi-hit game). Covariates that are
naturally per-batted-ball (batter_pitch_rv, pitcher_pitch_rv_allowed, since
they're pitch-type specific) get averaged across whatever's available;
covariates that are already game-constant (venue, weather, batter_tb_rate,
team quality) just take the first value. The "does this batter-game have
trustworthy inputs" drop happens AFTER aggregating, so it filters
observations, never the truth of the outcome itself.

WHAT'S REUSED, NOT DUPLICATED: same as calibrate_hits_model.py — load_batted_balls,
load_weather, get_primary_venues, categorize_wind, bucket_temp, bucket_speed
imported directly from calibrate_model.py.

MODEL: logistic regression, fit separately for L and R batters:
    is_2plus_tb ~ C(venue) + offense_quality + pitching_quality + batter_tb_rate
                  + batter_pitch_rv + pitcher_pitch_rv_allowed
                  + C(temp_bucket) + C(wind_dir) + C(speed_bucket)
Same significance rule, same reference-baseline-averaged-across-all-venues
approach as calibrate_model.py / calibrate_hits_model.py.

SETUP: pip install statsmodels
RUN:
    python3 calibrate_tb_model.py
    python3 calibrate_tb_model.py data/batted_balls_2025-04-01_2025-09-30.parquet
"""

import sys

import pandas as pd
import statsmodels.formula.api as smf

from historical_features import add_batter_tb_rate, pull_full_pitch_data, add_pitch_type_matchup, TOTAL_BASES_BY_EVENT
from calibrate_model import (
    load_batted_balls, load_weather, get_primary_venues,
    categorize_wind, bucket_temp, bucket_speed,
    MIN_TEAM_ROAD_SAMPLE,
)


def build_tb_dataset(batted: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    primary_venues = get_primary_venues(batted, weather)

    batted = add_batter_tb_rate(batted)

    start_dt = batted["game_date"].min().date().isoformat()
    end_dt = batted["game_date"].max().date().isoformat()
    pitches = pull_full_pitch_data(start_dt, end_dt)
    batted = add_pitch_type_matchup(pitches, batted)

    df = batted.merge(weather, on="game_pk", how="left")
    df["total_bases"] = df["events"].map(TOTAL_BASES_BY_EVENT).fillna(0).astype(int)
    df = df.dropna(subset=["venue", "stand", "home_team", "away_team", "inning_topbot"])
    df = df[df["venue"].isin(primary_venues)]

    df["condition_lower"] = df["condition"].astype(str).str.lower()
    df[["wind_speed", "wind_dir"]] = df["wind"].apply(lambda w: pd.Series(categorize_wind(w)))
    df["temp_bucket"] = df["temp"].apply(bucket_temp)
    df["speed_bucket"] = df["wind_speed"].apply(bucket_speed)

    df["batting_team"] = df["away_team"].where(df["inning_topbot"] == "Top", df["home_team"])
    df["pitching_team"] = df["home_team"].where(df["inning_topbot"] == "Top", df["away_team"])

    # road-only team-quality proxies, computed at the per-batted-ball level on total_bases
    # (a "how much damage does this team's contact do" measure) -- same circularity-avoidance
    # reasoning as HR/hits, just a TB-flavored metric instead of HR rate or hit rate.
    road_batting = df[df["batting_team"] == df["away_team"]]
    offense_quality = road_batting.groupby("batting_team")["total_bases"].agg(["mean", "count"])
    offense_quality = offense_quality[offense_quality["count"] >= MIN_TEAM_ROAD_SAMPLE]["mean"]

    road_pitching = df[df["pitching_team"] == df["away_team"]]
    pitching_quality = road_pitching.groupby("pitching_team")["total_bases"].agg(["mean", "count"])
    pitching_quality = pitching_quality[pitching_quality["count"] >= MIN_TEAM_ROAD_SAMPLE]["mean"]

    df["offense_quality"] = df["batting_team"].map(offense_quality)
    df["pitching_quality"] = df["pitching_team"].map(pitching_quality)

    print(f"Aggregating {len(df)} batted-ball rows to one row per batter-game "
          f"(2+ total bases is a whole-game fact, not a per-batted-ball one — see module docstring)...")
    game_level = df.groupby(["batter", "game_pk"], as_index=False).agg(
        total_bases_this_game=("total_bases", "sum"),
        stand=("stand", "first"),
        venue=("venue", "first"),
        temp_bucket=("temp_bucket", "first"),
        wind_dir=("wind_dir", "first"),
        speed_bucket=("speed_bucket", "first"),
        offense_quality=("offense_quality", "first"),
        pitching_quality=("pitching_quality", "first"),
        batter_tb_rate=("batter_tb_rate", "first"),
        batter_pitch_rv=("batter_pitch_rv", "mean"),
        pitcher_pitch_rv_allowed=("pitcher_pitch_rv_allowed", "mean"),
    )
    game_level["is_2plus_tb"] = (game_level["total_bases_this_game"] >= 2).astype(int)
    print(f"  {len(game_level)} batter-games. Overall 2+ TB rate: "
          f"{game_level['is_2plus_tb'].mean()*100:.1f}%.")

    before = len(game_level)
    game_level = game_level.dropna(subset=["batter_tb_rate", "batter_pitch_rv", "pitcher_pitch_rv_allowed",
                                            "offense_quality", "pitching_quality"])
    print(f"Dropped {before - len(game_level)} batter-games missing batter_tb_rate, matchup data, "
          f"or a reliable team-quality proxy (early season / too little history), {len(game_level)} remain.")

    return game_level


def fit_tb_model(df: pd.DataFrame, hand: str):
    sub = df[df["stand"] == hand].copy()
    print(f"\nFitting {hand}-handed batter TOTAL BASES model on {len(sub)} batter-games...")

    formula = ("is_2plus_tb ~ C(venue) + offense_quality + pitching_quality + batter_tb_rate "
               "+ batter_pitch_rv + pitcher_pitch_rv_allowed "
               "+ C(temp_bucket) + C(wind_dir) + C(speed_bucket)")
    model = smf.logit(formula, data=sub).fit(disp=0)

    avg_offense = sub["offense_quality"].mean()
    avg_pitching = sub["pitching_quality"].mean()
    avg_tb_rate = sub["batter_tb_rate"].mean()
    avg_batter_rv = sub["batter_pitch_rv"].mean()
    avg_pitcher_rv = sub["pitcher_pitch_rv_allowed"].mean()
    ref_temp = sub["temp_bucket"].mode()[0]
    ref_wind = sub["wind_dir"].mode()[0]
    ref_speed = sub["speed_bucket"].mode()[0]
    return model, sub, avg_offense, avg_pitching, avg_tb_rate, avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed


def extract_tb_park_factors(model, sub, hand, avg_offense, avg_pitching, avg_tb_rate,
                             avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed) -> pd.DataFrame:
    venues = sub["venue"].unique()
    base_covariates = pd.DataFrame({
        "offense_quality": [avg_offense], "pitching_quality": [avg_pitching], "batter_tb_rate": [avg_tb_rate],
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


def extract_tb_weather_factors(model, sub, hand, avg_offense, avg_pitching, avg_tb_rate,
                                avg_batter_rv, avg_pitcher_rv) -> pd.DataFrame:
    venues = sub["venue"].unique()
    combos = sub[["temp_bucket", "wind_dir", "speed_bucket"]].drop_duplicates()

    def avg_pred(temp, wind, speed):
        preds = []
        for v in venues:
            inp = pd.DataFrame({
                "venue": [v], "offense_quality": [avg_offense], "pitching_quality": [avg_pitching],
                "batter_tb_rate": [avg_tb_rate], "batter_pitch_rv": [avg_batter_rv],
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
        "batter_tb_rate": [avg_tb_rate], "batter_pitch_rv": [avg_batter_rv],
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
            "batter_tb_rate": [avg_tb_rate], "batter_pitch_rv": [avg_batter_rv],
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


def extract_batter_tbrate_factor(model, sub, hand, avg_offense, avg_pitching, avg_batter_rv,
                                  avg_pitcher_rv, ref_temp, ref_wind, ref_speed) -> pd.DataFrame:
    venues = sub["venue"].unique()
    p10, p50, p90 = sub["batter_tb_rate"].quantile([0.10, 0.50, 0.90])

    def avg_pred(rate_value):
        preds = []
        for v in venues:
            inp = pd.DataFrame({
                "venue": [v], "offense_quality": [avg_offense], "pitching_quality": [avg_pitching],
                "batter_tb_rate": [rate_value], "batter_pitch_rv": [avg_batter_rv],
                "pitcher_pitch_rv_allowed": [avg_pitcher_rv],
                "temp_bucket": [ref_temp], "wind_dir": [ref_wind], "speed_bucket": [ref_speed],
            })
            preds.append(model.predict(inp).iloc[0])
        return sum(preds) / len(preds)

    pred_low, pred_mid, pred_high = avg_pred(p10), avg_pred(p50), avg_pred(p90)

    ref_venue = venues[0]
    low_input = pd.DataFrame({"venue": [ref_venue], "offense_quality": [avg_offense],
                               "pitching_quality": [avg_pitching], "batter_tb_rate": [p10],
                               "batter_pitch_rv": [avg_batter_rv], "pitcher_pitch_rv_allowed": [avg_pitcher_rv],
                               "temp_bucket": [ref_temp], "wind_dir": [ref_wind], "speed_bucket": [ref_speed]})
    high_input = low_input.copy()
    high_input["batter_tb_rate"] = p90
    low_summary = model.get_prediction(low_input).summary_frame(alpha=0.05)
    high_summary = model.get_prediction(high_input).summary_frame(alpha=0.05)

    return pd.DataFrame([{
        "factor_type": "batter_tb_rate", "key": f"p10={p10:.3f} vs p90={p90:.3f} (median={p50:.3f})", "stand": hand,
        "pred_at_p10": pred_low, "pred_at_p50": pred_mid, "pred_at_p90": pred_high,
        "factor": pred_high / pred_low - 1,
        "factor_ci_low": high_summary["ci_lower"].iloc[0] / low_summary["ci_upper"].iloc[0] - 1,
        "factor_ci_high": high_summary["ci_upper"].iloc[0] / max(low_summary["ci_lower"].iloc[0], 1e-6) - 1,
        "significant": "yes" if low_summary["ci_upper"].iloc[0] < high_summary["ci_lower"].iloc[0] else "no (CIs overlap)",
    }])


def extract_tb_matchup_factor(model, sub, hand, avg_offense, avg_pitching, avg_tb_rate,
                               ref_temp, ref_wind, ref_speed) -> pd.DataFrame:
    venues = sub["venue"].unique()
    b_p10, b_p50, b_p90 = sub["batter_pitch_rv"].quantile([0.10, 0.50, 0.90])
    p_p10, p_p50, p_p90 = sub["pitcher_pitch_rv_allowed"].quantile([0.10, 0.50, 0.90])

    def avg_pred(batter_rv, pitcher_rv):
        preds = []
        for v in venues:
            inp = pd.DataFrame({
                "venue": [v], "offense_quality": [avg_offense], "pitching_quality": [avg_pitching],
                "batter_tb_rate": [avg_tb_rate], "batter_pitch_rv": [batter_rv],
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
                               "pitching_quality": [avg_pitching], "batter_tb_rate": [avg_tb_rate],
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

    df = build_tb_dataset(batted, weather)

    park_results, weather_results, tbrate_results, matchup_results = [], [], [], []
    for hand in ["L", "R"]:
        (model, sub, avg_offense, avg_pitching, avg_tb_rate, avg_batter_rv,
         avg_pitcher_rv, ref_temp, ref_wind, ref_speed) = fit_tb_model(df, hand)
        park_results.append(extract_tb_park_factors(model, sub, hand, avg_offense, avg_pitching, avg_tb_rate, avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed))
        weather_results.append(extract_tb_weather_factors(model, sub, hand, avg_offense, avg_pitching, avg_tb_rate, avg_batter_rv, avg_pitcher_rv))
        tbrate_results.append(extract_batter_tbrate_factor(model, sub, hand, avg_offense, avg_pitching, avg_batter_rv, avg_pitcher_rv, ref_temp, ref_wind, ref_speed))
        matchup_results.append(extract_tb_matchup_factor(model, sub, hand, avg_offense, avg_pitching, avg_tb_rate, ref_temp, ref_wind, ref_speed))

    park_df = pd.concat(park_results, ignore_index=True)
    weather_df = pd.concat(weather_results, ignore_index=True)
    tbrate_df = pd.concat(tbrate_results, ignore_index=True)
    matchup_df = pd.concat(matchup_results, ignore_index=True)

    print("\n" + "=" * 88)
    print("MODEL-BASED PARK FACTORS (TOTAL BASES, 2+)")
    print("=" * 88)
    for _, row in park_df.sort_values(["key", "stand"]).iterrows():
        print(f"  {row['key']:<28} {row['stand']}  factor={row['factor']:+.3f}  "
              f"CI=[{row['factor_ci_low']:+.3f}, {row['factor_ci_high']:+.3f}]  significant={row['significant']}")

    print("\n" + "=" * 88)
    print("MODEL-BASED WEATHER FACTORS (TOTAL BASES, 2+) (park, team quality, TB rate, matchup controlled for)")
    print("=" * 88)
    for _, row in weather_df.sort_values(["stand", "factor"], ascending=[True, False]).iterrows():
        print(f"  {row['key']:<38} {row['stand']}  n={row['n']:<5}  factor={row['factor']:+.3f}  "
              f"CI=[{row['factor_ci_low']:+.3f}, {row['factor_ci_high']:+.3f}]  significant={row['significant']}")

    print("\n" + "=" * 88)
    print("MODEL-BASED BATTER TB-RATE EFFECT (park, weather, team quality, matchup controlled for)")
    print("=" * 88)
    for _, row in tbrate_df.iterrows():
        print(f"  stand={row['stand']}  {row['key']}")
        print(f"    predicted 2+TB rate: cold={row['pred_at_p10']:.4f}  median={row['pred_at_p50']:.4f}  hot={row['pred_at_p90']:.4f}")
        print(f"    swing (hot vs cold): {row['factor']:+.3f}  CI=[{row['factor_ci_low']:+.3f}, {row['factor_ci_high']:+.3f}]  significant={row['significant']}")

    print("\n" + "=" * 88)
    print("MODEL-BASED PITCH-TYPE MATCHUP EFFECT ON 2+TB (park, weather, team quality, TB rate controlled for)")
    print("=" * 88)
    for _, row in matchup_df.iterrows():
        print(f"  stand={row['stand']}  {row['key']}")
        print(f"    predicted 2+TB rate: bad matchup={row['pred_bad']:.4f}  typical={row['pred_mid']:.4f}  good matchup={row['pred_good']:.4f}")
        print(f"    swing (good vs bad): {row['factor']:+.3f}  CI=[{row['factor_ci_low']:+.3f}, {row['factor_ci_high']:+.3f}]  significant={row['significant']}")

    park_df.to_csv("data/tb_park_factors_model.csv", index=False)
    weather_df.to_csv("data/tb_weather_factors_model.csv", index=False)
    tbrate_df.to_csv("data/tb_batter_tbrate_factor_model.csv", index=False)
    matchup_df.to_csv("data/tb_matchup_factor_model.csv", index=False)
    print(f"\nSaved to data/tb_park_factors_model.csv, data/tb_weather_factors_model.csv, "
          f"data/tb_batter_tbrate_factor_model.csv, data/tb_matchup_factor_model.csv")

    for label, results in [("park/hand", park_df), ("weather/hand", weather_df)]:
        n_sig = (results["significant"] == "yes").sum()
        print(f"{n_sig} of {len(results)} {label} effects are statistically significant even after "
              f"controlling for the other factors.")
