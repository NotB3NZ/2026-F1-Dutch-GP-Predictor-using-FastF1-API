

import logging

# pyrefly: ignore [missing-import]
import fastf1
import pandas as pd

from src.config import (
    CACHE_DIR,
    COMPLETED_ROUNDS,
    SEASON,
    SESSION_RACE,
    STARTING_GRID,
)

logger = logging.getLogger(__name__)


def init_cache():
    # Enable FastF1 API caching to the project's cache directory.
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    logger.info(f"FastF1 cache enabled at: {CACHE_DIR}")


def load_session(year: int, event, session_type: str):
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()
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

    all_results = []

    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        event_name = event["EventName"]

        # Only load completed rounds
        if round_num > max_rounds:
            logger.info(f"Skipping Round {round_num}: {event_name} (beyond max_rounds={max_rounds})")
            continue

        logger.info(f"Loading Round {round_num}: {event_name}...")

        session = load_session(year, round_num, SESSION_RACE)
        if session is None:
            continue

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


def get_starting_grid() -> pd.DataFrame:
   
    return pd.DataFrame(STARTING_GRID)
