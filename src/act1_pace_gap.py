"""
ACT 1: QUALIFYING VS RACE PACE GAP ANALYSIS (2026 SEASON)

Same approach as the validated 2023 project:
- PositionsGained = GridPosition - Position (positive = overperformed)
- DNFs EXCLUDED from pace gap averages but tracked as reliability metric
- Only main Sunday races (no sprints), so grid = qualifying position

Scoped to the 11 completed 2026 races (Australia → Hungary).
Output feeds into Act 3 as a feature/context input.

OUTPUTS:
- act1_avg_positions_gained.png  — Ranked bar chart of season averages
- act1_positions_gained_heatmap.png — Driver × Race heatmap
- act1_dnf_reliability.png — DNF rate by driver
- act1_results.csv — Season-level aggregated stats per driver
- act1_per_race.csv — Per-race detail (for Act 3 reuse)

Usage:
    python -m src.act1_pace_gap
"""

import logging
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    FIGURE_DPI,
    FIGURE_SIZE_TALL,
    FIGURE_SIZE_WIDE,
    OUTPUT_DIR,
    SEASON,
    COMPLETED_ROUNDS,
    TEAM_COLORS_2026,
)
from src.data_loader import load_all_race_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Data pipeline

def prepare_race_data(raw_results: pd.DataFrame) -> pd.DataFrame:
    """
    DATA CLEANING: Raw session results → Analytics-ready DataFrame.

    Steps:
    1. Select relevant columns
    2. Convert Position/GridPosition to numeric (errors → NaN)
    3. Flag DNFs (Position is NaN or 0, or Status contains retirement keywords)
    4. Compute PositionsGained for non-DNF rows

    Returns a DataFrame with one row per driver per race.
    """

    cols_to_keep = [
        "Abbreviation",
        "DriverNumber",
        "TeamName",
        "GridPosition",
        "Position",
        "Status",
        "RoundNumber",
        "EventName",
    ]

    available_cols = [c for c in cols_to_keep if c in raw_results.columns]
    df = raw_results[available_cols].copy()

    # Ensure numeric types
    df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")

    # Flag DNFs
    df["IsDNF"] = (
        df["Position"].isna()
        | (df["Position"] == 0)
        | df["Status"].str.contains(
            r"Retired|Accident|Collision|Spun off|Damage|Engine|Gearbox|"
            r"Hydraulic|Electrical|Mechanical|Withdrew|Disqualified|Excluded",
            case=False,
            na=False,
        )
    )

    # Compute positions gained
    # Positive = gained positions, Negative = lost positions
    df["PositionsGained"] = np.where(
        df["IsDNF"],
        np.nan,
        df["GridPosition"] - df["Position"],
    )

    return df


def aggregate_season_stats(per_race: pd.DataFrame) -> pd.DataFrame:
    # Aggregate per-race data into season-level stats per driver.

    stats = (
        per_race.groupby(["Abbreviation", "TeamName"])
        .agg(
            AvgPositionsGained=("PositionsGained", "mean"),
            StdPositionsGained=("PositionsGained", "std"),
            MedianPositionsGained=("PositionsGained", "median"),
            TotalPositionsGained=("PositionsGained", "sum"),
            RacesCompleted=("IsDNF", lambda x: (~x).sum()),
            TotalRaces=("IsDNF", "count"),
            DNFCount=("IsDNF", "sum"),
        )
        .reset_index()
    )

    stats["DNFRate"] = (stats["DNFCount"] / stats["TotalRaces"] * 100).round(1)
    stats["CompletionRate"] = (
        stats["RacesCompleted"] / stats["TotalRaces"] * 100
    ).round(1)

    # Also compute average grid and average finish for Act 3 features
    grid_avg = (
        per_race.groupby("Abbreviation")["GridPosition"]
        .mean()
        .rename("AvgGridPosition")
    )
    finish_avg = (
        per_race[~per_race["IsDNF"]]
        .groupby("Abbreviation")["Position"]
        .mean()
        .rename("AvgFinishPosition")
    )
    grid_std = (
        per_race.groupby("Abbreviation")["GridPosition"]
        .std()
        .fillna(0)
        .rename("GridVariance")
    )

    stats = stats.merge(grid_avg, on="Abbreviation", how="left")
    stats = stats.merge(finish_avg, on="Abbreviation", how="left")
    stats = stats.merge(grid_std, on="Abbreviation", how="left")

    # Sort by average positions gained
    stats = stats.sort_values("AvgPositionsGained", ascending=False).reset_index(
        drop=True
    )

    return stats



# Visualization

def _get_team_color(team_name: str) -> str:
    #Get team color with partial matching.
    for key, color in TEAM_COLORS_2026.items():
        if key.lower() in team_name.lower() or team_name.lower() in key.lower():
            return color
    return "#888888"


def plot_avg_positions_gained(stats: pd.DataFrame) -> None:
    # Butterfly bar chart of average positions gained per driver.

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_TALL)

    colors = [_get_team_color(team) for team in stats["TeamName"]]

    ax.barh(
        stats["Abbreviation"],
        stats["AvgPositionsGained"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    # Zero line
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")

    # Annotations: races completed / total and DNFs
    for i, (_, row) in enumerate(stats.iterrows()):
        completed = int(row["RacesCompleted"])
        total = int(row["TotalRaces"])
        dnf_count = int(row["DNFCount"])
        avg = row["AvgPositionsGained"]

        if avg >= 0:
            x_pos = avg + 0.15
            ha = "left"
        else:
            x_pos = avg - 0.15
            ha = "right"

        label = f"{avg:+.1f}  ({completed}/{total} races"
        if dnf_count > 0:
            label += f", {dnf_count} DNF"
        label += ")"

        ax.text(
            x_pos, i, label,
            va="center", ha=ha, fontsize=7, color="#333333",
        )

    ax.set_xlabel("Average Positions Gained (Qualifying → Race)", fontsize=11)
    ax.set_title(
        f"{SEASON} Season (Rounds 1–{COMPLETED_ROUNDS}) — Qualifying vs Race Performance\n"
        f"Positive = Overperformed (gained positions), Negative = Underperformed\n"
        f"DNFs excluded from averages — race counts shown per driver",
        fontsize=12,
        fontweight="bold",
    )

    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()

    path = OUTPUT_DIR / "act1_avg_positions_gained.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close(fig)


def plot_positions_heatmap(per_race: pd.DataFrame) -> None:
    # Heatmap of positions gained/lost per driver per race.

    event_labels = (
        per_race.drop_duplicates("RoundNumber")
        .sort_values("RoundNumber")
        .set_index("RoundNumber")["EventName"]
    )

    pivot = per_race.pivot_table(
        index="Abbreviation",
        columns="RoundNumber",
        values="PositionsGained",
        aggfunc="first",
    )

    # Best driver by season average at the top
    driver_order = (
        per_race.groupby("Abbreviation")["PositionsGained"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    pivot = pivot.reindex(driver_order)

    # Abbreviated event names
    short_names = {
        r: name.replace(" Grand Prix", "").replace(" GP", "")[:12]
        for r, name in event_labels.items()
    }
    pivot = pivot.rename(columns=short_names)

    fig, ax = plt.subplots(figsize=(16, 12))  # Taller for 22 drivers

    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Positions Gained / Lost"},
        mask=pivot.isna(),
    )

    # Mark DNFs explicitly
    dnf_pivot = per_race.pivot_table(
        index="Abbreviation",
        columns="RoundNumber",
        values="IsDNF",
        aggfunc="first",
    ).reindex(driver_order).rename(columns=short_names)

    for i in range(dnf_pivot.shape[0]):
        for j in range(dnf_pivot.shape[1]):
            if dnf_pivot.iloc[i, j] == True:  # noqa: E712
                ax.text(
                    j + 0.5, i + 0.5, "DNF",
                    ha="center", va="center",
                    fontsize=6, color="#CC0000", fontweight="bold",
                )

    ax.set_title(
        f"{SEASON} Season (Rounds 1–{COMPLETED_ROUNDS}) — Positions Gained/Lost Per Race\n"
        "Green = Gained positions | Red = Lost positions | DNF = Did Not Finish",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Race", fontsize=11)
    ax.set_ylabel("Driver", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)

    plt.tight_layout()

    path = OUTPUT_DIR / "act1_positions_gained_heatmap.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close(fig)


def plot_dnf_reliability(stats: pd.DataFrame) -> None:
    # DNF rate bar chart showing driver reliability.

    df = stats.sort_values("DNFRate", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    colors = [_get_team_color(team) for team in df["TeamName"]]

    ax.barh(
        df["Abbreviation"],
        df["DNFRate"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    for i, (_, row) in enumerate(df.iterrows()):
        dnf_count = int(row["DNFCount"])
        total = int(row["TotalRaces"])
        ax.text(
            row["DNFRate"] + 0.5, i,
            f"{dnf_count} DNF out of {total} races",
            va="center", ha="left", fontsize=9, color="#333333",
        )

    ax.set_xlabel("DNF Rate (%)", fontsize=11)
    ax.set_title(
        f"{SEASON} Season (Rounds 1–{COMPLETED_ROUNDS}) — Driver Reliability (DNF Rate)\n"
        "Higher = more retirements (mechanical failures, accidents, etc.)",
        fontsize=12,
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(df["DNFRate"].max() + 10, 25))

    plt.tight_layout()

    path = OUTPUT_DIR / "act1_dnf_reliability.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close(fig)



# Data exports

def export_csvs(stats: pd.DataFrame, per_race: pd.DataFrame) -> None:
    """Export season stats and per-race data for Act 3 consumption."""

    stats_path = OUTPUT_DIR / "act1_results.csv"
    stats.to_csv(stats_path, index=False)
    logger.info(f"Saved: {stats_path}")

    per_race_path = OUTPUT_DIR / "act1_per_race.csv"
    per_race.to_csv(per_race_path, index=False)
    logger.info(f"Saved: {per_race_path}")


# Main

def main():
    # Run the full Act 1 pipeline.
    print(f"\n{'='*60}")
    print(f"  ACT 1 — Qualifying vs Race Pace Gap ({SEASON} Season, Rounds 1–{COMPLETED_ROUNDS})")
    print(f"{'='*60}\n")

    # Step 1: Load all race results
    print("Step 1/4: Loading race results (this may take a while on first run)...")
    raw_results = load_all_race_results(SEASON, max_rounds=COMPLETED_ROUNDS)
    print(f"  → Loaded {len(raw_results)} result rows\n")

    # Step 2: Prepare and clean data
    print("Step 2/4: Preparing data...")
    per_race = prepare_race_data(raw_results)
    total_dnfs = per_race["IsDNF"].sum()
    total_entries = len(per_race)
    print(f"  → {total_entries} entries, {total_dnfs} DNFs ({total_dnfs/total_entries*100:.1f}%)\n")

    # Step 3: Aggregate season stats
    print("Step 3/4: Aggregating season statistics...")
    stats = aggregate_season_stats(per_race)

    # Print a quick summary table
    print("\n  Top 5 Overperformers:")
    for _, row in stats.head(5).iterrows():
        print(
            f"    {row['Abbreviation']:>3} ({row['TeamName']:<20}): "
            f"{row['AvgPositionsGained']:+.2f} avg  "
            f"({int(row['RacesCompleted'])}/{int(row['TotalRaces'])} races, "
            f"{int(row['DNFCount'])} DNFs)"
        )

    print("\n  Top 5 Underperformers:")
    for _, row in stats.tail(5).iterrows():
        print(
            f"    {row['Abbreviation']:>3} ({row['TeamName']:<20}): "
            f"{row['AvgPositionsGained']:+.2f} avg  "
            f"({int(row['RacesCompleted'])}/{int(row['TotalRaces'])} races, "
            f"{int(row['DNFCount'])} DNFs)"
        )

    # Step 4: Generate visualizations and export
    print("\n\nStep 4/4: Generating charts and exporting data...")
    plot_avg_positions_gained(stats)
    plot_positions_heatmap(per_race)
    plot_dnf_reliability(stats)
    export_csvs(stats, per_race)

    print(f"\n{'='*60}")
    print("  Act 1 complete! Outputs saved to outputs/")
    print(f"{'='*60}")
    print(f"\n  Charts:")
    print(f"    • outputs/act1_avg_positions_gained.png")
    print(f"    • outputs/act1_positions_gained_heatmap.png")
    print(f"    • outputs/act1_dnf_reliability.png")
    print(f"  Data:")
    print(f"    • outputs/act1_results.csv")
    print(f"    • outputs/act1_per_race.csv")
    print()

    return stats, per_race


if __name__ == "__main__":
    stats, per_race = main()
