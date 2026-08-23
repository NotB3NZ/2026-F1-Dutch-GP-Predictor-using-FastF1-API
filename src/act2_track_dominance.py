"""
ACT 2: ZANDVOORT TRACK DOMINANCE MAP

Adapted from the 2023 project's act2_dominance_map.py with these changes:
- Evaluates the current 2026 weekend sessions (FP1, Q) purely on 2026 data
- Produces a dominance map + speed trace per session
- Generates a sector-level summary

Default comparison: NOR vs VER (pole-sitter vs Zandvoort specialist)

OUTPUTS:
- act2_track_dominance_{D1}_vs_{D2}_{SESSION}.png — Track map per session
- act2_speed_comparison_{D1}_vs_{D2}_{SESSION}.png — Speed trace per session
- act2_sector_summary.csv — Sector breakdown across sessions

Usage:
    python -m src.act2_track_dominance
"""

import logging
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from src.config import (
    FIGURE_DPI,
    OUTPUT_DIR,
    ZANDVOORT_SESSIONS,
    TEAM_COLORS_2026,
    SEASON,
)
from src.data_loader import init_cache, load_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# DRIVER CONFIGS
# ----------------------------------------------------------

DRIVER_1 = "NOR"  # Pole position holder
DRIVER_2 = "RUS"  # Zandvoort specialist (won 2023)
EVENT_NAME = "Dutch Grand Prix"



# Telemetry extraction and processing

def get_fastest_lap_telemetry(session, driver_abbrev: str):
    #Extract telemetry from a driver's fastest lap.

    try:
        driver_laps = session.laps.pick_drivers(driver_abbrev)
        fastest = driver_laps.pick_fastest()

        if fastest is None or (hasattr(fastest, "empty") and fastest.empty):
            logger.warning(f"No fastest lap found for {driver_abbrev}")
            return None, None

        telemetry = fastest.get_telemetry().add_distance()

        if telemetry.empty or "Distance" not in telemetry.columns:
            logger.warning(f"No telemetry data for {driver_abbrev}")
            return None, None

        logger.info(
            f"{driver_abbrev} fastest lap: {fastest['LapTime']} "
            f"({len(telemetry)} telemetry points)"
        )
        return telemetry, fastest

    except Exception as e:
        logger.error(f"Error getting telemetry for {driver_abbrev}: {e}")
        return None, None


def compute_dominance(tel1, tel2, n_points: int = 500):

    max_dist = min(tel1["Distance"].max(), tel2["Distance"].max())
    distance = np.linspace(0, max_dist, n_points)

    speed1 = np.interp(distance, tel1["Distance"], tel1["Speed"])
    speed2 = np.interp(distance, tel2["Distance"], tel2["Speed"])

    x = np.interp(distance, tel1["Distance"], tel1["X"])
    y = np.interp(distance, tel1["Distance"], tel1["Y"])

    dominance = speed1 - speed2

    return {
        "distance": distance,
        "speed1": speed1,
        "speed2": speed2,
        "x": x,
        "y": y,
        "dominance": dominance,
    }


def compute_sector_breakdown(data: dict, driver1: str, driver2: str):

    n = len(data["dominance"])
    sector_size = n // 3
    sectors = {}

    for i, sector_name in enumerate(["S1", "S2", "S3"]):
        start = i * sector_size
        end = (i + 1) * sector_size if i < 2 else n
        sector_dom = data["dominance"][start:end]

        d1_faster_pct = (sector_dom > 0).sum() / len(sector_dom) * 100
        avg_delta = sector_dom.mean()

        sectors[sector_name] = {
            "d1_faster_pct": round(d1_faster_pct, 1),
            "d2_faster_pct": round(100 - d1_faster_pct, 1),
            "avg_speed_delta": round(avg_delta, 1),
            "dominant_driver": driver1 if d1_faster_pct > 50 else driver2,
        }

    return sectors



# Visualization

def get_team_color(session, driver_abbrev: str, fallback: str = "#888888") -> str:
    """Get team color from session results, with fallback."""
    try:
        driver_info = session.results[
            session.results["Abbreviation"] == driver_abbrev
        ]
        if not driver_info.empty and "TeamColor" in driver_info.columns:
            color = driver_info.iloc[0]["TeamColor"]
            if color and isinstance(color, str):
                return f"#{color}" if not color.startswith("#") else color
    except Exception:
        pass

    try:
        if not driver_info.empty:
            team = driver_info.iloc[0].get("TeamName", "")
            for key, color in TEAM_COLORS_2026.items():
                if key.lower() in str(team).lower():
                    return color
    except Exception:
        pass

    return fallback


def plot_track_dominance(
    data: dict, driver1: str, driver2: str,
    color1: str, color2: str, event_name: str,
    session_type: str, year: int,
    lap_time_1=None, lap_time_2=None,
):
    #Plot the track map color-coded by driver dominance.

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    points = np.array([data["x"], data["y"]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "dominance", [color2, "#FFFFFF", color1], N=256,
    )

    max_delta = max(abs(data["dominance"].min()), abs(data["dominance"].max()), 1)
    norm = plt.Normalize(-max_delta, max_delta)

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=4)
    lc.set_array(data["dominance"][:-1])
    ax.add_collection(lc)

    cbar = fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(
        f"← {driver2} faster  |  {driver1} faster →",
        fontsize=11, color="white", labelpad=10,
    )
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    title = f"{year} {event_name} — {session_type} Track Dominance\n"
    title += f"{driver1} vs {driver2}"
    if lap_time_1 is not None and lap_time_2 is not None:
        title += f"\n{driver1}: {lap_time_1}  |  {driver2}: {lap_time_2}"

    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=15)

    ax.set_aspect("equal")
    ax.axis("off")

    # Start/finish
    ax.plot(
        data["x"][0], data["y"][0],
        marker="o", markersize=10, color="white",
        zorder=5, label="Start/Finish",
    )
    ax.annotate(
        "S/F", (data["x"][0], data["y"][0]),
        textcoords="offset points", xytext=(10, 10),
        fontsize=10, color="white", fontweight="bold",
    )

    # Driver name labels
    ax.text(
        0.02, 0.02, f"■ {driver1}",
        transform=ax.transAxes, fontsize=14, fontweight="bold",
        color=color1, va="bottom",
    )
    ax.text(
        0.98, 0.02, f"{driver2} ■",
        transform=ax.transAxes, fontsize=14, fontweight="bold",
        color=color2, va="bottom", ha="right",
    )

    plt.tight_layout()

    filename = f"act2_track_dominance_{driver1}_vs_{driver2}_{session_type}.png"
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved: {path}")
    plt.close(fig)

    return path


def plot_speed_comparison(
    data: dict, driver1: str, driver2: str,
    color1: str, color2: str, event_name: str,
    session_type: str, year: int,
):
    #Plot speed traces and dominance shading.

    fig, (ax_speed, ax_delta) = plt.subplots(
        2, 1, figsize=(16, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("#1a1a2e")

    dist_km = data["distance"] / 1000

    # Speed traces
    ax_speed.set_facecolor("#16213e")
    ax_speed.plot(
        dist_km, data["speed1"], color=color1, linewidth=1.5,
        label=driver1, alpha=0.9,
    )
    ax_speed.plot(
        dist_km, data["speed2"], color=color2, linewidth=1.5,
        label=driver2, alpha=0.9,
    )
    ax_speed.set_ylabel("Speed (km/h)", fontsize=11, color="white")
    ax_speed.legend(
        fontsize=12, loc="upper right", facecolor="#16213e",
        edgecolor="gray", labelcolor="white",
    )

    ax_speed.set_title(
        f"{year} {event_name} — {session_type} Speed Comparison\n"
        f"{driver1} vs {driver2}",
        fontsize=14, fontweight="bold", color="white", pad=15,
    )
    ax_speed.tick_params(colors="white")
    ax_speed.grid(alpha=0.2, color="white")

    # Speed delta with fill
    ax_delta.set_facecolor("#16213e")
    ax_delta.fill_between(
        dist_km, data["dominance"], 0,
        where=data["dominance"] > 0,
        color=color1, alpha=0.6, label=f"{driver1} faster",
    )
    ax_delta.fill_between(
        dist_km, data["dominance"], 0,
        where=data["dominance"] < 0,
        color=color2, alpha=0.6, label=f"{driver2} faster",
    )
    ax_delta.axhline(y=0, color="white", linewidth=0.5, alpha=0.5)
    ax_delta.set_ylabel("Δ Speed (km/h)", fontsize=11, color="white")
    ax_delta.set_xlabel("Distance (km)", fontsize=11, color="white")
    ax_delta.legend(
        fontsize=10, loc="upper right", facecolor="#16213e",
        edgecolor="gray", labelcolor="white",
    )
    ax_delta.tick_params(colors="white")
    ax_delta.grid(alpha=0.2, color="white")

    plt.tight_layout()

    filename = f"act2_speed_comparison_{driver1}_vs_{driver2}_{session_type}.png"
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved: {path}")
    plt.close(fig)

    return path



# Main

def main(
    driver1: str = DRIVER_1,
    driver2: str = DRIVER_2,
    event: str = EVENT_NAME,
):
    #Run the full Act 2 pipeline across 2026 Zandvoort sessions.

    print(f"\n{'='*60}")
    print(f"  ACT 2 — Zandvoort Track Dominance Map")
    print(f"  {driver1} vs {driver2} | {SEASON} {event}")
    print(f"  Sessions: {ZANDVOORT_SESSIONS}")
    print(f"{'='*60}\n")

    init_cache()

    all_sector_data = []
    year = SEASON

    for session_type in ZANDVOORT_SESSIONS:
        print(f"\n--- {year} {event} {session_type} ---")

        # Load session
        print(f"  Loading {year} {event} {session_type}...")
        session = load_session(year, event, session_type)
        if session is None:
            print(f"  ⚠ Could not load {session_type} session. Skipping.")
            continue

        # Extract telemetry
        print(f"  Extracting fastest lap telemetry...")
        tel1, lap1 = get_fastest_lap_telemetry(session, driver1)
        tel2, lap2 = get_fastest_lap_telemetry(session, driver2)

        if tel1 is None or tel2 is None:
            print(f"  ⚠ Missing telemetry for one/both drivers in {session_type}. Skipping.")
            continue

        # Format lap times
        def _fmt_laptime(lt):
            if lt is None:
                return "N/A"
            total_seconds = lt.total_seconds()
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            return f"{minutes}:{seconds:06.3f}"

        lap_time_1 = _fmt_laptime(lap1["LapTime"]) if lap1 is not None else "N/A"
        lap_time_2 = _fmt_laptime(lap2["LapTime"]) if lap2 is not None else "N/A"
        print(f"  {driver1} fastest: {lap_time_1}")
        print(f"  {driver2} fastest: {lap_time_2}")

        # Compute dominance
        print(f"  Computing track dominance...")
        data = compute_dominance(tel1, tel2, n_points=500)

        # Quick stats
        d1_faster = (data["dominance"] > 0).sum()
        d2_faster = (data["dominance"] < 0).sum()
        total = len(data["dominance"])
        print(f"  {driver1} faster in {d1_faster}/{total} segments ({d1_faster/total*100:.0f}%)")
        print(f"  {driver2} faster in {d2_faster}/{total} segments ({d2_faster/total*100:.0f}%)")

        # Sector breakdown
        sectors = compute_sector_breakdown(data, driver1, driver2)
        for sector_name, sector_data in sectors.items():
            all_sector_data.append({
                "Year": year,
                "Session": session_type,
                "Sector": sector_name,
                "Driver1": driver1,
                "Driver2": driver2,
                f"{driver1}_FasterPct": sector_data["d1_faster_pct"],
                f"{driver2}_FasterPct": sector_data["d2_faster_pct"],
                "AvgSpeedDelta_kmh": sector_data["avg_speed_delta"],
                "DominantDriver": sector_data["dominant_driver"],
            })

        # Get colors
        color1 = get_team_color(session, driver1, fallback="#FF8000")
        color2 = get_team_color(session, driver2, fallback="#3671C6")
        if color1 == color2:
            color2 = "#FFD700"

        # Generate plots
        print(f"  Generating visualizations...")
        plot_track_dominance(
            data, driver1, driver2, color1, color2,
            event, session_type, year, lap_time_1, lap_time_2,
        )
        plot_speed_comparison(
            data, driver1, driver2, color1, color2,
            event, session_type, year,
        )

    # Export sector summary
    if all_sector_data:
        sector_df = pd.DataFrame(all_sector_data)
        sector_path = OUTPUT_DIR / "act2_sector_summary.csv"
        sector_df.to_csv(sector_path, index=False)
        logger.info(f"Saved: {sector_path}")

        print(f"\n{'='*60}")
        print(f"  Sector Summary (across all years)")
        print(f"{'='*60}")
        print(sector_df.to_string(index=False))
    else:
        print("\n⚠ No sector data generated — check telemetry availability.")

    print(f"\n{'='*60}")
    print(f"  Act 2 complete! Outputs saved to outputs/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
