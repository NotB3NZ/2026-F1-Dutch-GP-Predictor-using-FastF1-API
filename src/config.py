
# Configuration for the 2026 Singapore GP Predictor.

from pathlib import Path


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "2026_singapore_gp"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

for d in [CACHE_DIR, OUTPUT_DIR, NOTEBOOKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# Season and event configuration

SEASON = 2026
TARGET_EVENT = "Singapore Grand Prix"
TARGET_ROUND = 17
COMPLETED_ROUNDS = 16  # Australia → Bahrain; Singapore race results are excluded.

# Only the target qualifying session is used for Act 2.
TARGET_SESSIONS = ["Q"]

# Session type identifiers used by FastF1
SESSION_RACE = "R"
SESSION_QUALIFYING = "Q"
SESSION_SPRINT = "S"
SESSION_FP1 = "FP1"
SESSION_FP2 = "FP2"
SESSION_FP3 = "FP3"


# Pre-race grid adjustments; driver identities and qualifying positions come
# from the cached Singapore Q session, never from Singapore race results.
# Source: https://www.formula1.com/en/results/2026/races/1296/singapore/starting-grid
GRID_SOURCE = "https://www.formula1.com/en/results/2026/races/1296/singapore/starting-grid"
STARTING_ORDER = [
    "VER", "LEC", "HAM", "ANT", "NOR", "PIA", "GAS", "LAW", "LIN",
    "HUL", "BEA", "OCO", "SAI", "BOR", "COL", "ALB", "PER", "BOT",
    "RUS", "ALO", "STR", "HAD",
]
PIT_LANE_STARTERS = {"ALO", "STR", "HAD"}

RACE_DAY_NOTES = {
    "weather": "Weather is not a model input; heat, humidity and rain remain uncertainties.",
    "track_character": "Marina Bay street circuit; the model has no circuit-specific feature.",
    "qualifying": "Verstappen qualified first, ahead of Leclerc and Hamilton.",
    "grid_penalties": (
        "Russell received a 60-place power-unit penalty. Alonso, Stroll and Hadjar "
        "are pit-lane starters, represented as effective grid positions 20–22. "
        "Hadjar was permitted to start without a qualifying time."
    ),
}


# 2026 Team colors (for plotting)

TEAM_COLORS_2026 = {
    "Mercedes":          "#27F4D2",
    "Ferrari":           "#E8002D",
    "McLaren":           "#FF8000",
    "Red Bull Racing":   "#3671C6",
    "Aston Martin":      "#229971",
    "Alpine":            "#FF87BC",
    "Williams":          "#1868DB",
    "Racing Bulls":      "#6692FF",
    "Haas F1 Team":      "#B6BABD",
    "Audi":              "#FF0000",
    "Cadillac":          "#1B3D2F",
}



# F1 points system

POINTS_MAP = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1,
}



# Plotting defaults

FIGURE_DPI = 150
FIGURE_SIZE_WIDE = (14, 8)
FIGURE_SIZE_SQUARE = (10, 10)
FIGURE_SIZE_TALL = (12, 18)  # Slightly taller than 2023 to fit 22 drivers
