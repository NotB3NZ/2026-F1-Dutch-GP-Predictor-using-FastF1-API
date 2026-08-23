
# Configuration for the 2026 Dutch GP Predictor.

from pathlib import Path


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

for d in [CACHE_DIR, OUTPUT_DIR, NOTEBOOKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# Season and event configuration

SEASON = 2026
TARGET_EVENT = "Dutch Grand Prix"
TARGET_ROUND = 12
COMPLETED_ROUNDS = 11  # Australia → Hungary (rounds 1–11 are finished)

# 2026 Zandvoort sessions for Act 2 track dominance analysis
ZANDVOORT_SESSIONS = ["FP1", "FP2", "FP3", "Q"]

# Session type identifiers used by FastF1
SESSION_RACE = "R"
SESSION_QUALIFYING = "Q"
SESSION_SPRINT = "S"
SESSION_FP1 = "FP1"
SESSION_FP2 = "FP2"
SESSION_FP3 = "FP3"


# 2026 Dutch GP Starting Grid (post-qualifying, no penalties applied)

# Source: FIA qualifying classification, August 22, 2026.
# Hamilton/Verstappen had lap times deleted for track limits (T1/T3),
# but NO grid penalties were issued. Grid = qualifying order.

# Lawson/Tsunoda swap: Lawson promoted to Red Bull (Hadjar injured),
# Tsunoda slots into Racing Bulls alongside Lindblad.

STARTING_GRID = [
    {"grid": 1,  "driver": "NOR", "full_name": "Lando Norris",       "team": "McLaren"},
    {"grid": 2,  "driver": "RUS", "full_name": "George Russell",     "team": "Mercedes"},
    {"grid": 3,  "driver": "ANT", "full_name": "Kimi Antonelli",     "team": "Mercedes"},
    {"grid": 4,  "driver": "PIA", "full_name": "Oscar Piastri",      "team": "McLaren"},
    {"grid": 5,  "driver": "HAM", "full_name": "Lewis Hamilton",     "team": "Ferrari"},
    {"grid": 6,  "driver": "LEC", "full_name": "Charles Leclerc",    "team": "Ferrari"},
    {"grid": 7,  "driver": "VER", "full_name": "Max Verstappen",     "team": "Red Bull Racing"},
    {"grid": 8,  "driver": "LAW", "full_name": "Liam Lawson",        "team": "Red Bull Racing"},
    {"grid": 9,  "driver": "BOR", "full_name": "Gabriel Bortoleto",  "team": "Audi"},
    {"grid": 10, "driver": "LIN", "full_name": "Arvid Lindblad",     "team": "Racing Bulls"},
    {"grid": 11, "driver": "GAS", "full_name": "Pierre Gasly",       "team": "Alpine"},
    {"grid": 12, "driver": "TSU", "full_name": "Yuki Tsunoda",       "team": "Racing Bulls"},
    {"grid": 13, "driver": "HUL", "full_name": "Nico Hulkenberg",    "team": "Audi"},
    {"grid": 14, "driver": "COL", "full_name": "Franco Colapinto",   "team": "Alpine"},
    {"grid": 15, "driver": "OCO", "full_name": "Esteban Ocon",       "team": "Haas F1 Team"},
    {"grid": 16, "driver": "ALB", "full_name": "Alex Albon",         "team": "Williams"},
    {"grid": 17, "driver": "SAI", "full_name": "Carlos Sainz",       "team": "Williams"},
    {"grid": 18, "driver": "ALO", "full_name": "Fernando Alonso",    "team": "Aston Martin"},
    {"grid": 19, "driver": "STR", "full_name": "Lance Stroll",       "team": "Aston Martin"},
    {"grid": 20, "driver": "BEA", "full_name": "Oliver Bearman",     "team": "Haas F1 Team"},
    {"grid": 21, "driver": "BOT", "full_name": "Valtteri Bottas",    "team": "Cadillac"},
    {"grid": 22, "driver": "PER", "full_name": "Sergio Pérez",       "team": "Cadillac"},
]

# Driver → Team mapping for Round 12 specifically

DRIVER_TEAM_MAP_R12 = {entry["driver"]: entry["team"] for entry in STARTING_GRID}


# Race-day context flags 

RACE_DAY_NOTES = {
    "weather": "Dry, 15-18°C, cloudy with sunny spells. No rain expected.",
    "antonelli_damage": (
        "Floor damage from Sprint Qualifying gravel excursion. "
        "Team patched but ~25pts downforce loss. Handling compromised in "
        "low-speed corners."
    ),
    "lawson_swap": (
        "Liam Lawson promoted to Red Bull Racing (replacing injured Hadjar). "
        "Yuki Tsunoda moved to Racing Bulls for this event."
    ),
    "final_zandvoort": "Last Dutch GP under current contract. Farewell race for the venue.",
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
