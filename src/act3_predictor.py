"""
ACT 3: 2026 SINGAPORE GRAND PRIX RACE PREDICTOR

Trains a Random Forest + Linear Regression ensemble on the 2026 season 
(Rounds 1-16) to predict the finishing order for the Singapore GP (Round 17).

Usage:
    .venv/bin/python -m src.act3_predictor
"""

import logging
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    COMPLETED_ROUNDS,
    FIGURE_DPI,
    FIGURE_SIZE_WIDE,
    OUTPUT_DIR,
    POINTS_MAP,
    RACE_DAY_NOTES,
    SEASON,
    TARGET_EVENT,
    GRID_SOURCE,
    TEAM_COLORS_2026,
)
from src.data_loader import get_starting_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")



# Feature engineering

FEATURE_COLS = [
    "GridPosition",
    "DriverHistoricalGain",
    "TeamRollingPoints",
    "DriverCompletionRate",
    "DriverAvgGrid",
    "GridVariance",
    "TeamDNFRate",
]


def load_act1_data():
    #Load the per-race and season stats data from Act 1's output.
    per_race_path = OUTPUT_DIR / "act1_per_race.csv"
    stats_path = OUTPUT_DIR / "act1_results.csv"

    if not per_race_path.exists() or not stats_path.exists():
        raise FileNotFoundError(
            f"Act 1 data not found. Run Act 1 first: python -m src.act1_pace_gap"
        )

    per_race = pd.read_csv(per_race_path)
    stats = pd.read_csv(stats_path)
    expected_rounds = set(range(1, COMPLETED_ROUNDS + 1))
    if set(per_race["RoundNumber"]) != expected_rounds:
        raise ValueError("Act 1 data must contain exactly rounds 1–16; rerun Act 1")
    if per_race.duplicated(["RoundNumber", "Abbreviation"]).any():
        raise ValueError("Duplicate driver/race rows in Act 1 data")
    logger.info(f"Loaded Act 1 data: {len(per_race)} per-race rows, {len(stats)} driver stats")
    return per_race, stats


def build_training_features(per_race: pd.DataFrame) -> pd.DataFrame:
    # Engineer features for each driver-race in the training data (rounds 1–16).
    df = per_race.copy()
    if not df["RoundNumber"].between(1, COMPLETED_ROUNDS).all():
        raise ValueError("Training data contains a round outside the pre-Singapore cutoff")

    df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df["RoundNumber"] = pd.to_numeric(df["RoundNumber"], errors="coerce")

    df = df.sort_values(["Abbreviation", "RoundNumber"]).reset_index(drop=True)

    features = []

    for driver in df["Abbreviation"].unique():
        driver_df = df[df["Abbreviation"] == driver].copy()

        for idx, row in driver_df.iterrows():
            round_num = row["RoundNumber"]
            prior = driver_df[driver_df["RoundNumber"] < round_num]
            grid_pos = row["GridPosition"]

            if len(prior) == 0:
                hist_gain = 0.0
                completion_rate = 100.0
                avg_grid = grid_pos
                grid_var = 0.0
            else:
                prior_non_dnf = prior[prior["IsDNF"] == False]  # noqa: E712
                hist_gain = (
                    prior_non_dnf["PositionsGained"].mean()
                    if len(prior_non_dnf) > 0
                    else 0.0
                )
                completion_rate = (
                    (prior["IsDNF"] == False).sum() / len(prior) * 100  # noqa: E712
                )
                avg_grid = prior["GridPosition"].mean()
                grid_var = prior["GridPosition"].std()
                if pd.isna(grid_var):
                    grid_var = 0.0

            # Team rolling points
            team = row["TeamName"]
            team_prior = df[
                (df["TeamName"] == team) & (df["RoundNumber"] < round_num)
            ]
            team_points = team_prior["Position"].apply(
                lambda p: POINTS_MAP.get(int(p), 0) if pd.notna(p) and p > 0 else 0
            ).sum()

            # Team DNF rate
            team_all = df[
                (df["TeamName"] == team) & (df["RoundNumber"] < round_num)
            ]
            team_dnf_rate = (
                team_all["IsDNF"].mean() * 100 if len(team_all) > 0 else 0.0
            )

            features.append({
                "Abbreviation": row["Abbreviation"],
                "TeamName": team,
                "RoundNumber": round_num,
                "EventName": row["EventName"],
                "GridPosition": grid_pos,
                "DriverHistoricalGain": hist_gain,
                "TeamRollingPoints": team_points,
                "DriverCompletionRate": completion_rate,
                "DriverAvgGrid": avg_grid,
                "GridVariance": grid_var,
                "TeamDNFRate": team_dnf_rate,
                # Target
                "FinishPosition": row["Position"],
                "IsDNF": row["IsDNF"],
            })

    features_df = pd.DataFrame(features)

    # Drop DNFs and non-starters from training
    features_df = features_df[features_df["IsDNF"] == False].copy()  # noqa: E712

    # Drop rows with NaN in critical columns
    critical_cols = FEATURE_COLS
    features_df = features_df.dropna(subset=critical_cols + ["FinishPosition"])

    logger.info(f"Built training feature matrix: {features_df.shape}")
    return features_df


def build_prediction_features(
    stats: pd.DataFrame,
    per_race: pd.DataFrame,
) -> pd.DataFrame:
    # Build feature vector for each driver in the race.
    # Uses the FULL 16-race season data.
    grid = get_starting_grid()
    pred_features = []

    for _, entry in grid.iterrows():
        abbr = entry["driver"]
        team = entry["team"]
        grid_pos = entry["grid"]

        # Look up driver in Act 1 stats
        driver_stats = stats[stats["Abbreviation"] == abbr]

        if not driver_stats.empty:
            row = driver_stats.iloc[0]
            hist_gain = row.get("AvgPositionsGained", 0.0)
            completion_rate = row.get("CompletionRate", 100.0)
            avg_grid = row.get("AvgGridPosition", grid_pos)
            grid_var = row.get("GridVariance", 0.0)
        else:
            # Driver not in Act 1 stats (e.g., Hadjar missed 3 races)
            # Use reasonable defaults
            hist_gain = 0.0
            completion_rate = 100.0
            avg_grid = grid_pos
            grid_var = 0.0

        # Team rolling points (full season total heading into R17)
        team_data = per_race[per_race["TeamName"].str.contains(team, case=False, na=False)]
        if team_data.empty:
            # Try partial match
            for t in per_race["TeamName"].unique():
                if team.lower() in t.lower() or t.lower() in team.lower():
                    team_data = per_race[per_race["TeamName"] == t]
                    break

        team_points = 0
        if not team_data.empty:
            team_points = team_data["Position"].apply(
                lambda p: POINTS_MAP.get(int(p), 0) if pd.notna(p) and p > 0 else 0
            ).sum()

        # Team DNF rate
        team_dnf_rate = 0.0
        if not team_data.empty:
            team_dnf_rate = team_data["IsDNF"].mean() * 100

        pred_features.append({
            "Abbreviation": abbr,
            "FullName": entry["full_name"],
            "TeamName": team,
            "GridPosition": grid_pos,
            "QualifyingPosition": int(entry["qualifying_position"]),
            "PitLaneStart": bool(entry["pit_lane_start"]),
            "DriverHistoricalGain": hist_gain if pd.notna(hist_gain) else 0.0,
            "TeamRollingPoints": team_points,
            "DriverCompletionRate": completion_rate,
            "DriverAvgGrid": avg_grid if pd.notna(avg_grid) else grid_pos,
            "GridVariance": grid_var if pd.notna(grid_var) else 0.0,
            "TeamDNFRate": team_dnf_rate,
        })

    pred_df = pd.DataFrame(pred_features)

    logger.info(f"Built prediction feature matrix: {pred_df.shape}")
    return pred_df



# Model training and prediction
def train_model(features_df: pd.DataFrame):
    # Train Random Forest on all training data and return the fitted model.
    # Evaluate the ensemble on each round using only earlier rounds.
    
    X = features_df[FEATURE_COLS].values
    y = features_df["FinishPosition"].values
    groups = features_df["RoundNumber"].values

    # Walk-forward validation prevents later races (and their rolling history)
    # from leaking into the validation of an earlier race. R1 has no prior fold.
    cv_predictions = np.full(len(y), np.nan)
    for test_round in sorted(np.unique(groups))[1:]:
        train_idx = groups < test_round
        test_idx = groups == test_round
        rf_cv = RandomForestRegressor(
            n_estimators=200, max_depth=8,
            min_samples_split=5, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
        rf_cv.fit(X[train_idx], y[train_idx])
        lr_cv = LinearRegression().fit(X[train_idx], y[train_idx])
        cv_predictions[test_idx] = (
            np.clip(rf_cv.predict(X[test_idx]), 1, 22)
            + np.clip(lr_cv.predict(X[test_idx]), 1, 22)
        ) / 2

    valid = np.isfinite(cv_predictions)
    cv_rounds = groups[valid]
    cv_predictions = cv_predictions[valid]
    y_cv = y[valid]
    cv_mae = mean_absolute_error(y_cv, cv_predictions)
    cv_rmse = np.sqrt(mean_squared_error(y_cv, cv_predictions))
    cv_r2 = r2_score(y_cv, cv_predictions)

    # Podium accuracy
    actual_podium = (y_cv <= 3).astype(int)
    pred_podium = (cv_predictions <= 3.5).astype(int)
    podium_acc = (actual_podium == pred_podium).mean() * 100

    within_1 = np.mean(np.abs(y_cv - cv_predictions) <= 1) * 100
    within_3 = np.mean(np.abs(y_cv - cv_predictions) <= 3) * 100

    cv_metrics = {
        "MAE": cv_mae,
        "RMSE": cv_rmse,
        "R²": cv_r2,
        "Within ±1 pos (%)": within_1,
        "Within ±3 pos (%)": within_3,
        "Podium classification accuracy (%)": podium_acc,
    }

    logger.info(
        f"Walk-forward CV: MAE={cv_mae:.2f}, RMSE={cv_rmse:.2f}, R²={cv_r2:.3f}, "
        f"Podium={podium_acc:.0f}%"
    )

    # Train model on ALL data
    rf_final = RandomForestRegressor(
        n_estimators=200, max_depth=8,
        min_samples_split=5, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )
    rf_final.fit(X, y)

    # Also train a Linear Regression baseline
    lr_final = LinearRegression()
    lr_final.fit(X, y)

    return {
        "rf_model": rf_final,
        "lr_model": lr_final,
        "cv_metrics": cv_metrics,
        "cv_predictions": cv_predictions,
        "cv_actuals": y_cv,
        "cv_rounds": cv_rounds,
        "feature_importances": rf_final.feature_importances_,
    }


def generate_race_prediction(model_data: dict, pred_features: pd.DataFrame):
    # Generate the final race prediction for the Singapore GP.

    X_pred = pred_features[FEATURE_COLS].values

    # Random Forest prediction (primary)
    rf_pred = model_data["rf_model"].predict(X_pred)
    rf_pred = np.clip(rf_pred, 1, 22)

    # Linear Regression prediction (comparison)
    lr_pred = model_data["lr_model"].predict(X_pred)
    lr_pred = np.clip(lr_pred, 1, 22)

    # Create prediction DataFrame
    prediction = pred_features[[
        "Abbreviation", "FullName", "TeamName", "GridPosition",
        "QualifyingPosition", "PitLaneStart",
    ]].copy()
    prediction["RF_PredictedPos"] = rf_pred
    prediction["LR_PredictedPos"] = lr_pred

    # Average of RF and LR
    prediction["EnsemblePredictedPos"] = (rf_pred + lr_pred) / 2

    # Rank to get a clean finishing order (no duplicate positions)
    prediction["PredictedFinish"] = (
        prediction["EnsemblePredictedPos"].rank(method="first").astype(int)
    )

    # Sort by the final rank.
    prediction = prediction.sort_values("PredictedFinish").reset_index(drop=True)

    # DNF%
    prediction["TeamDNFRate"] = pred_features.set_index("Abbreviation").loc[
        prediction["Abbreviation"].values, "TeamDNFRate"
    ].values

    return prediction



# Visualization

def _get_team_color(team_name: str) -> str:
    # Get team color with partial matching.
    for key, color in TEAM_COLORS_2026.items():
        if key.lower() in team_name.lower() or team_name.lower() in key.lower():
            return color
    return "#888888"


def plot_predicted_finishing_order(prediction: pd.DataFrame) -> None:
    # Horizontal bar chart showing predicted finishing order.

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    colors = [_get_team_color(team) for team in prediction["TeamName"]]

    y_pos = range(len(prediction))
    bars = ax.barh(
        y_pos,
        prediction["GridPosition"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.4,
        label="Grid Position",
    )

    # Overlay predicted position as markers
    for i, (_, row) in enumerate(prediction.iterrows()):
        # Arrow from grid to predicted
        grid = row["GridPosition"]
        pred = row["PredictedFinish"]
        change = grid - pred  # positive = gained places

        label = f"P{pred} (Grid P{int(grid)}, {change:+d})"
        color = colors[i]

        ax.text(
            0.5, i, f"  P{pred}  {row['Abbreviation']} ({row['TeamName']})",
            va="center", ha="left", fontsize=10, color="white", fontweight="bold",
        )

        ax.text(
            22, i, label,
            va="center", ha="right", fontsize=9, color="#aaaaaa",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"P{row['PredictedFinish']}" for _, row in prediction.iterrows()],
        fontsize=10, color="white",
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 23)
    ax.set_xlabel("Grid Position", fontsize=11, color="white")
    ax.set_title(
        f"{SEASON} {TARGET_EVENT} — Predicted Finishing Order\n"
        f"Model: Random Forest + Linear Regression Ensemble | Trained on Rounds 1–{COMPLETED_ROUNDS}",
        fontsize=14, fontweight="bold", color="white", pad=15,
    )
    ax.tick_params(colors="white")
    ax.grid(axis="x", alpha=0.15, color="white")

    # Podium highlight
    for i in range(min(3, len(prediction))):
        ax.get_yticklabels()[i].set_color("#FFD700")
        ax.get_yticklabels()[i].set_fontweight("bold")

    plt.tight_layout()

    path = OUTPUT_DIR / "act3_singapore_gp_prediction.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved: {path}")
    plt.close(fig)


def plot_feature_importance(model_data: dict) -> None:
    # Bar chart of Random Forest feature importances.

    importances = model_data["feature_importances"]
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    feature_names = [FEATURE_COLS[i] for i in indices]
    feature_vals = importances[indices]

    ax.barh(
        range(len(feature_names)), feature_vals,
        color="#00d4ff", edgecolor="white", linewidth=0.5,
    )

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=11, color="white")
    ax.invert_yaxis()

    for i, v in enumerate(feature_vals):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=10, color="white")

    ax.set_xlabel("Importance", fontsize=11, color="white")
    ax.set_title(
        f"Random Forest — Feature Importance\n"
        f"What matters most for predicting finishing position?",
        fontsize=13, fontweight="bold", color="white",
    )
    ax.tick_params(colors="white")
    ax.grid(axis="x", alpha=0.15, color="white")

    plt.tight_layout()

    path = OUTPUT_DIR / "act3_feature_importance.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved: {path}")
    plt.close(fig)


def plot_training_performance(model_data: dict) -> None:
    # Plot for predicted vs actual for training confidence.

    y_actual = model_data["cv_actuals"]
    y_pred = model_data["cv_predictions"]

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.plot([1, 22], [1, 22], "w--", alpha=0.4, linewidth=1, label="Perfect")
    ax.fill_between(
        [1, 22], [1 - 3, 22 - 3], [1 + 3, 22 + 3],
        alpha=0.1, color="white", label="±3 positions",
    )
    ax.scatter(
        y_actual, y_pred,
        alpha=0.6, s=30, c="#00d4ff", edgecolor="white", linewidth=0.3,
    )

    metrics = model_data["cv_metrics"]
    ax.text(
        0.05, 0.95,
        f"MAE: {metrics['MAE']:.2f}\n"
        f"RMSE: {metrics['RMSE']:.2f}\n"
        f"R²: {metrics['R²']:.3f}\n"
        f"Podium classification: {metrics['Podium classification accuracy (%)']:.0f}%",
        transform=ax.transAxes, fontsize=11,
        va="top", color="white",
        bbox=dict(boxstyle="round", facecolor="#16213e", edgecolor="gray"),
    )

    ax.set_xlabel("Actual Position", fontsize=11, color="white")
    ax.set_ylabel("Predicted Position", fontsize=11, color="white")
    ax.set_title(
        f"Walk-Forward Ensemble Validation\n"
        f"Training on {SEASON} Rounds 1–{COMPLETED_ROUNDS}",
        fontsize=13, fontweight="bold", color="white",
    )
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 23)
    ax.tick_params(colors="white")
    ax.grid(alpha=0.15, color="white")
    ax.legend(
        fontsize=9, facecolor="#16213e", edgecolor="gray",
        labelcolor="white", loc="lower right",
    )

    plt.tight_layout()

    path = OUTPUT_DIR / "act3_training_performance.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved: {path}")
    plt.close(fig)


# Report generation

def generate_report(prediction: pd.DataFrame, model_data: dict) -> str:
    # Generate a plain-language prediction report.

    lines = []
    lines.append("=" * 70)
    lines.append("  🏁 2026 SINGAPORE GRAND PRIX — RACE PREDICTION REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Model: Random Forest + Linear Regression Ensemble")
    lines.append(f"  Training: {SEASON} Season, Rounds 1–{COMPLETED_ROUNDS} ({COMPLETED_ROUNDS} races)")
    lines.append("  Prediction inputs: Singapore Q + published pre-race grid adjustments")
    lines.append("  Singapore R, S and SQ are excluded from training and prediction inputs.")
    lines.append("  Retrospective forecast generated after the event, using pre-race inputs.")
    lines.append(f"  Grid source: {GRID_SOURCE}")
    lines.append("=" * 70)
    lines.append("")

    # Predicted podium
    lines.append("PREDICTED PODIUM")
    lines.append("-" * 70)
    podium = prediction.head(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, (_, row) in enumerate(podium.iterrows()):
        change = int(row["GridPosition"]) - row["PredictedFinish"]
        lines.append(
            f"  {medals[i]} P{row['PredictedFinish']}: {row['FullName']} "
            f"({row['TeamName']}) — Grid P{int(row['GridPosition'])} ({change:+d})"
        )
    lines.append("")

    # Full predicted order
    lines.append("FULL PREDICTED FINISHING ORDER")
    lines.append("-" * 70)
    for _, row in prediction.iterrows():
        change = int(row["GridPosition"]) - row["PredictedFinish"]
        flag = ""
        if row.get("TeamDNFRate", 0) > 15:
            flag = " ⚠ DNF risk"
        lines.append(
            f"  P{row['PredictedFinish']:>2}: {row['FullName']:<22} "
            f"({row['TeamName']:<18}) "
            f"Grid P{int(row['GridPosition']):>2} ({change:+d}){flag}"
        )
    lines.append("")

    # Model confidence
    lines.append("MODEL CONFIDENCE (Walk-Forward Ensemble Validation)")
    lines.append("-" * 70)
    cv = model_data["cv_metrics"]
    lines.append(f"  MAE:  {cv['MAE']:.2f} positions (average prediction error)")
    lines.append(f"  RMSE: {cv['RMSE']:.2f} positions")
    lines.append(f"  R²:   {cv['R²']:.3f}")
    lines.append(f"  Within ±1 position: {cv['Within ±1 pos (%)']:.0f}%")
    lines.append(f"  Within ±3 positions: {cv['Within ±3 pos (%)']:.0f}%")
    lines.append(f"  Podium/non-podium classification accuracy: {cv['Podium classification accuracy (%)']:.0f}%")
    lines.append("")

    # Race-day context
    lines.append("RACE-DAY CONTEXT & UNCERTAINTY FACTORS")
    lines.append("-" * 70)
    lines.append(f"  Weather: {RACE_DAY_NOTES['weather']}")
    lines.append("")
    lines.append(f"  Track character: {RACE_DAY_NOTES['track_character']}")
    lines.append("")
    lines.append(f"  Qualifying: {RACE_DAY_NOTES['qualifying']}")
    lines.append("")
    lines.append(f"  Grid adjustments: {RACE_DAY_NOTES['grid_penalties']}")
    lines.append("")
    lines.append("LIMITATIONS")
    lines.append("-" * 70)
    lines.append("  • Training excludes DNFs and non-starters; predicted ranks assume drivers finish normally.")
    lines.append("  • Team DNF rates are historical context, not individual DNF predictions.")
    lines.append("  • Weather, safety cars, strategy and track-specific effects are not modeled.")
    lines.append("  • Pit-lane starts use effective positions 20–22, without a separate time penalty.")
    lines.append("  • Validation trains on earlier rounds only, testing rounds 2–16.")
    lines.append("    R1 has no earlier training fold; all 16 races feed the final model.")
    lines.append("  • Podium classification includes non-podium drivers and is not a probability")
    lines.append("    of predicting the full podium. Validation scores are continuous positions;")
    lines.append("    the final finishing order ranks the ensemble scores across all drivers.")
    lines.append("")
    report = "\n".join(lines)
    return report



# Main

def main():
    # Run the full Act 3 pipeline: train, predict, visualize, report.

    print(f"\n{'='*60}")
    print(f"  ACT 3 — 2026 Singapore Grand Prix Race Predictor")
    print(f"{'='*60}\n")

    # Step 1: Load Act 1 data
    print("Step 1/5: Loading Act 1 per-race data...")
    per_race, stats = load_act1_data()
    print(f"  → {len(per_race)} driver-race entries loaded\n")

    # Step 2: Build features
    print("Step 2/5: Engineering features...")
    train_features = build_training_features(per_race)
    print(f"  → Training matrix: {train_features.shape}")

    pred_features = build_prediction_features(stats, per_race)
    print(f"  → Prediction matrix: {pred_features.shape}\n")

    train_features.to_csv(OUTPUT_DIR / "act3_training_features.csv", index=False)
    pred_features.to_csv(OUTPUT_DIR / "act3_prediction_features.csv", index=False)

    # Step 3: Train model
    print("Step 3/5: Training model (with walk-forward CV)...")
    model_data = train_model(train_features)
    pd.DataFrame({
        "RoundNumber": model_data["cv_rounds"],
        "ActualPosition": model_data["cv_actuals"],
        "PredictedPosition": model_data["cv_predictions"],
    }).to_csv(OUTPUT_DIR / "act3_validation_predictions.csv", index=False)

    # Step 4: Generate prediction
    print("Step 4/5: Generating race prediction...")
    prediction = generate_race_prediction(model_data, pred_features)

    # Step 5: Visualize and export
    print("Step 5/5: Generating visualizations and report...")
    plot_predicted_finishing_order(prediction)
    plot_feature_importance(model_data)
    plot_training_performance(model_data)

    report = generate_report(prediction, model_data)
    report_path = OUTPUT_DIR / "act3_prediction_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved: {report_path}")

    pred_path = OUTPUT_DIR / "act3_predictions.csv"
    prediction.to_csv(pred_path, index=False)
    logger.info(f"Saved: {pred_path}")

    # Print quick stats to terminal
    print("\n\n" + "="*70)
    print("  QUICK STATS")
    print("="*70 + "\n")
    report_lines = report.split("\n")
    start_idx = report_lines.index("PREDICTED PODIUM")
    end_idx = report_lines.index("RACE-DAY CONTEXT & UNCERTAINTY FACTORS")
    print("\n".join(report_lines[start_idx:end_idx]))

    print(f"\n{'='*60}")
    print(f"  Act 3 complete! Outputs saved to outputs/2026_singapore_gp/")
    print(f"{'='*60}")
    print(f"\n  Charts:")
    print(f"    • outputs/2026_singapore_gp/act3_singapore_gp_prediction.png")
    print(f"    • outputs/2026_singapore_gp/act3_feature_importance.png")
    print(f"    • outputs/2026_singapore_gp/act3_training_performance.png")
    print(f"  Data:")
    print(f"    • outputs/2026_singapore_gp/act3_predictions.csv")
    print(f"    • outputs/2026_singapore_gp/act3_prediction_report.txt")
    print()

    return prediction, model_data


if __name__ == "__main__":
    prediction, model_data = main()
