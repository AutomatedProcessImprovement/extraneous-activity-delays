import os
from dataclasses import dataclass
from pathlib import Path

from estimate_start_times.config import EventLogIDs, DEFAULT_CSV_IDS


def get_project_dir() -> Path:
    return Path(os.path.dirname(__file__)).parent.parent


@dataclass
class DurationDistribution:
    type: str = "NORMAL"
    mean: str = "NaN"  # Warning! this value is always interpreted as seconds
    arg1: str = "NaN"
    arg2: str = "NaN"
    rawMean: str = "NaN"
    rawArg1: str = "NaN"
    rawArg2: str = "NaN"
    unit: str = "seconds"  # This is the unit to show in the interface by transforming the values in seconds


@dataclass
class Configuration:
    """Class storing the configuration parameters for the start time estimation.

    Attributes:
        log_ids                     Identifiers for each key element (e.g. executed activity or resource).
    """
    log_ids: EventLogIDs = DEFAULT_CSV_IDS

    PATH_PROJECT = get_project_dir()
    PATH_INPUTS = PATH_PROJECT.joinpath("event_logs")
    PATH_OUTPUTS = PATH_PROJECT.joinpath("outputs")
    PATH_SIMULATED = PATH_OUTPUTS.joinpath("simulated_logs")
    PATH_EXTERNAL_TOOLS = PATH_PROJECT.joinpath("external_tools")
    PATH_BIMP = PATH_PROJECT.joinpath("external_tools").joinpath("simulator").joinpath("qbp-simulator-engine.jar")
