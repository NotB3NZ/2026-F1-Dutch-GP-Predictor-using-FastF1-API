
# Configuration for the 2026 Bahrain GP (Sepang) Predictor.

from pathlib import Path


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "2026_bahrain_gp"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

for d in [CACHE_DIR, OUTPUT_DIR, NOTEBOOKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# Season and event configuration

SEASON = 2026
TARGET_EVENT = "Bahrain Grand Prix"
TARGET_ROUND = 16
COMPLETED_ROUNDS = 15  # Australia → Azerbaijan GP (rounds 1–15 are finished)

# 2026 Bahrain sessions for Act 2 track dominance analysis
BAHRAIN_SESSIONS = ["FP1", "FP2", "FP3", "Q"]

# Session type identifiers used by FastF1
SESSION_RACE = "R"
SESSION_QUALIFYING = "Q"
SESSION_SPRINT = "S"
SESSION_FP1 = "FP1"
SESSION_FP2 = "FP2"
SESSION_FP3 = "FP3"


# 2026 Bahrain GP Starting Grid (from qualifying + grid penalties)

# Source: FIA race classification grid positions, October 2026.
# Verstappen on pole (1:35.130), Hamilton P2, Antonelli P3.
# Hadjar initially P3 but dropped to P8 due to power unit penalty.
# Colapinto P21 (15-place penalty: 5 for Baku collision, 10 for PU).
# Lindblad P22 (30-place penalty for PU elements).

STARTING_GRID = [
    {"grid": 1,  "driver": "VER", "full_name": "Max Verstappen",       "team": "Red Bull Racing"},
    {"grid": 2,  "driver": "HAM", "full_name": "Lewis Hamilton",       "team": "Ferrari"},
    {"grid": 3,  "driver": "ANT", "full_name": "Kimi Antonelli",       "team": "Mercedes"},
    {"grid": 4,  "driver": "LEC", "full_name": "Charles Leclerc",      "team": "Ferrari"},
    {"grid": 5,  "driver": "NOR", "full_name": "Lando Norris",         "team": "McLaren"},
    {"grid": 6,  "driver": "PIA", "full_name": "Oscar Piastri",        "team": "McLaren"},
    {"grid": 7,  "driver": "RUS", "full_name": "George Russell",       "team": "Mercedes"},
    {"grid": 8,  "driver": "HAD", "full_name": "Isack Hadjar",         "team": "Red Bull Racing"},
    {"grid": 9,  "driver": "GAS", "full_name": "Pierre Gasly",         "team": "Alpine"},
    {"grid": 10, "driver": "BOR", "full_name": "Gabriel Bortoleto",    "team": "Audi"},
    {"grid": 11, "driver": "LAW", "full_name": "Liam Lawson",          "team": "Racing Bulls"},
    {"grid": 12, "driver": "ALO", "full_name": "Fernando Alonso",      "team": "Aston Martin"},
    {"grid": 13, "driver": "SAI", "full_name": "Carlos Sainz",         "team": "Williams"},
    {"grid": 14, "driver": "STR", "full_name": "Lance Stroll",         "team": "Aston Martin"},
    {"grid": 15, "driver": "HUL", "full_name": "Nico Hulkenberg",      "team": "Audi"},
    {"grid": 16, "driver": "BEA", "full_name": "Oliver Bearman",       "team": "Haas F1 Team"},
    {"grid": 17, "driver": "OCO", "full_name": "Esteban Ocon",         "team": "Haas F1 Team"},
    {"grid": 18, "driver": "ALB", "full_name": "Alexander Albon",      "team": "Williams"},
    {"grid": 19, "driver": "BOT", "full_name": "Valtteri Bottas",      "team": "Cadillac"},
    {"grid": 20, "driver": "PER", "full_name": "Sergio Pérez",         "team": "Cadillac"},
    {"grid": 21, "driver": "COL", "full_name": "Franco Colapinto",     "team": "Alpine"},
    {"grid": 22, "driver": "LIN", "full_name": "Arvid Lindblad",       "team": "Racing Bulls"},
]

# Driver → Team mapping for Round 16 specifically

DRIVER_TEAM_MAP_R16 = {entry["driver"]: entry["team"] for entry in STARTING_GRID}


# Race-day context flags 

RACE_DAY_NOTES = {
    "weather": (
        "Hot and humid conditions expected at Sepang. High chance of "
        "late afternoon thunderstorms, typical for the Malaysian climate."
    ),
    "track_character": (
        "The Sepang International Circuit is a high-speed, flowing track "
        "with long straights and fast sweeping corners. Tire degradation "
        "is expected to be very high due to the abrasive surface and high "
        "track temperatures. The two long straights back-to-back offer "
        "excellent overtaking opportunities."
    ),
    "verstappen_pole": (
        "Max Verstappen claims his first pole position of the 2026 season, "
        "putting Red Bull in a strong position on a track that demands aero efficiency."
    ),
    "grid_penalties": (
        "Isack Hadjar dropped from P3 to P8 due to PU penalties. Colapinto "
        "and Lindblad start at the back after heavy penalties for PU changes "
        "and collisions."
    ),
    "location_change": (
        "The Bahrain Grand Prix was relocated to the Sepang International "
        "Circuit in Malaysia due to regional tensions in the Middle East."
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
