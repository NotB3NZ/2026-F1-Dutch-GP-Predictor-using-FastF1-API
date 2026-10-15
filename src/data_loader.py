


import logging

# pyrefly: ignore [missing-import]
import fastf1
import pandas as pd

from src.config import (
    CACHE_DIR,
    COMPLETED_ROUNDS,
    SEASON,
    SESSION_RACE,
    TARGET_EVENT,
    TARGET_ROUND,
    SESSION_QUALIFYING,
    OUTPUT_DIR,
    STARTING_ORDER,
    PIT_LANE_STARTERS,
)

logger = logging.getLogger(__name__)


def init_cache():
    # Enable FastF1 API caching to the project's cache directory.
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    logger.info(f"FastF1 cache enabled at: {CACHE_DIR}")


def load_session(year: int, event, session_type: str, **load_options):
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load(**load_options)
        if session.results.empty:
            raise RuntimeError("Session has no driver results")
        logger.info(f"Loaded: {year} {event} {session_type}")
        return session
    except Exception as e:
        logger.warning(f"Failed to load {year} {event} {session_type}: {e}")
        return None


def load_all_race_results(
    year: int = SEASON,
    max_rounds: int = COMPLETED_ROUNDS,
) -> pd.DataFrame:
    
    init_cache()

    schedule = fastf1.get_event_schedule(year, include_testing=False)

    expected_rounds = set(range(1, max_rounds + 1))
    scheduled_rounds = set(schedule["RoundNumber"].astype(int))
    if not expected_rounds <= scheduled_rounds:
        raise RuntimeError("Schedule is missing required training rounds")

    all_results = []

    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        event_name = event["EventName"]

        # Only load completed rounds
        if round_num not in expected_rounds:
            logger.info(f"Skipping Round {round_num}: {event_name} (beyond max_rounds={max_rounds})")
            continue

        logger.info(f"Loading Round {round_num}: {event_name}...")

        session = load_session(
            year, round_num, SESSION_RACE,
            laps=False, telemetry=False, weather=False, messages=False,
        )
        if session is None:
            raise RuntimeError(f"Required training race unavailable: R{round_num} {event_name}")
        if session.results["Position"].isna().any() or session.results["Status"].eq("").any():
            raise RuntimeError(f"Incomplete results for R{round_num} {event_name}")

        results = session.results.copy()

        # Add round metadata
        results["RoundNumber"] = round_num
        results["EventName"] = event_name

        all_results.append(results)

    if not all_results:
        raise RuntimeError(f"No race results loaded for {year}. Check network/cache.")

    combined = pd.concat(all_results, ignore_index=True)
    logger.info(
        f"Loaded {len(all_results)} races, "
        f"{len(combined)} total result rows for {year}."
    )

    return combined


def load_target_qualifying():
    """Cache and validate target Q, without loading the target race or sprint."""
    init_cache()
    session = load_session(SEASON, TARGET_EVENT, SESSION_QUALIFYING)
    if session is None or int(session.event["RoundNumber"]) != TARGET_ROUND:
        raise RuntimeError("Target qualifying is unavailable or has the wrong round")
    results = session.results
    if (results["Abbreviation"].duplicated().any()
            or set(results["Abbreviation"]) != set(STARTING_ORDER)
            or results["Position"].isna().any()
            or results["Q1"].notna().sum() == 0):
        raise RuntimeError("Target qualifying results are incomplete")
    return session


def get_starting_grid() -> pd.DataFrame:
    session = load_target_qualifying()
    qualifying = session.results[[
        "Position", "Abbreviation", "FullName", "TeamName", "Q1", "Q2", "Q3",
    ]].sort_values("Position").copy()
    qualifying.to_csv(OUTPUT_DIR / "qualifying_results.csv", index=False)
    grid = qualifying.rename(columns={
        "Position": "qualifying_position", "Abbreviation": "driver",
        "FullName": "full_name", "TeamName": "team",
    })
    order = {driver: pos for pos, driver in enumerate(STARTING_ORDER, start=1)}
    grid["grid"] = grid["driver"].map(order).astype(int)
    grid["pit_lane_start"] = grid["driver"].isin(PIT_LANE_STARTERS)
    grid = grid.sort_values("grid").reset_index(drop=True)
    grid.to_csv(OUTPUT_DIR / "starting_grid.csv", index=False)
    return grid
