
# Configuration for the 2026 Spanish GP (Barcelona) Predictor.

from pathlib import Path


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "2026_spanish_gp"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

for d in [CACHE_DIR, OUTPUT_DIR, NOTEBOOKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# Season and event configuration

SEASON = 2026
TARGET_EVENT = "Spanish Grand Prix"
TARGET_ROUND = 14
COMPLETED_ROUNDS = 13  # Australia → Italian GP (rounds 1–13 are finished)

# 2026 Barcelona sessions for Act 2 track dominance analysis
BARCELONA_SESSIONS = ["FP1", "FP2", "FP3", "Q"]

# Session type identifiers used by FastF1
SESSION_RACE = "R"
SESSION_QUALIFYING = "Q"
SESSION_SPRINT = "S"
SESSION_FP1 = "FP1"
SESSION_FP2 = "FP2"
SESSION_FP3 = "FP3"


# 2026 Spanish GP Starting Grid (from race session grid positions)

# Source: FIA race classification grid positions, September 2026.
# Norris on pole, Antonelli P2, Verstappen P3.
# Bearman (P21) and Stroll (P22) did not set qualifying times — start from back.

STARTING_GRID = [
    {"grid": 1,  "driver": "NOR", "full_name": "Lando Norris",        "team": "McLaren"},
    {"grid": 2,  "driver": "ANT", "full_name": "Kimi Antonelli",      "team": "Mercedes"},
    {"grid": 3,  "driver": "VER", "full_name": "Max Verstappen",      "team": "Red Bull Racing"},
    {"grid": 4,  "driver": "HAM", "full_name": "Lewis Hamilton",      "team": "Ferrari"},
    {"grid": 5,  "driver": "LEC", "full_name": "Charles Leclerc",     "team": "Ferrari"},
    {"grid": 6,  "driver": "RUS", "full_name": "George Russell",      "team": "Mercedes"},
    {"grid": 7,  "driver": "PIA", "full_name": "Oscar Piastri",       "team": "McLaren"},
    {"grid": 8,  "driver": "LAW", "full_name": "Liam Lawson",         "team": "Red Bull Racing"},
    {"grid": 9,  "driver": "COL", "full_name": "Franco Colapinto",    "team": "Alpine"},
    {"grid": 10, "driver": "LIN", "full_name": "Arvid Lindblad",      "team": "Racing Bulls"},
    {"grid": 11, "driver": "HUL", "full_name": "Nico Hulkenberg",     "team": "Audi"},
    {"grid": 12, "driver": "BOR", "full_name": "Gabriel Bortoleto",   "team": "Audi"},
    {"grid": 13, "driver": "OCO", "full_name": "Esteban Ocon",        "team": "Haas F1 Team"},
    {"grid": 14, "driver": "GAS", "full_name": "Pierre Gasly",        "team": "Alpine"},
    {"grid": 15, "driver": "TSU", "full_name": "Yuki Tsunoda",        "team": "Racing Bulls"},
    {"grid": 16, "driver": "ALB", "full_name": "Alexander Albon",     "team": "Williams"},
    {"grid": 17, "driver": "SAI", "full_name": "Carlos Sainz",        "team": "Williams"},
    {"grid": 18, "driver": "ALO", "full_name": "Fernando Alonso",     "team": "Aston Martin"},
    {"grid": 19, "driver": "PER", "full_name": "Sergio Pérez",        "team": "Cadillac"},
    {"grid": 20, "driver": "BOT", "full_name": "Valtteri Bottas",     "team": "Cadillac"},
    {"grid": 21, "driver": "BEA", "full_name": "Oliver Bearman",      "team": "Haas F1 Team"},
    {"grid": 22, "driver": "STR", "full_name": "Lance Stroll",        "team": "Aston Martin"},
]

# Driver → Team mapping for Round 14 specifically

DRIVER_TEAM_MAP_R14 = {entry["driver"]: entry["team"] for entry in STARTING_GRID}


# Race-day context flags 

RACE_DAY_NOTES = {
    "weather": (
        "Late summer conditions in Barcelona. Warm and dry, typical "
        "Mediterranean weather with temperatures around 28-30°C."
    ),
    "track_character": (
        "Circuit de Barcelona-Catalunya is a well-known track with a mix of "
        "high-speed and technical sections. Sector 1 features the long main straight "
        "and Turn 1 braking zone, while Sector 3 has the demanding final chicane. "
        "DRS zones on the main straight and back straight offer overtaking opportunities, "
        "though the track is generally harder to overtake on than Monza."
    ),
    "bearman_stroll_dns": (
        "Oliver Bearman (Haas) and Lance Stroll (Aston Martin) did not set "
        "qualifying times and start from the back of the grid (P21, P22)."
    ),
    "norris_pole": (
        "Lando Norris takes pole position for the Spanish Grand Prix, "
        "continuing McLaren's strong qualifying form."
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
