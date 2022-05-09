import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from estimate_start_times.config import EventLogIDs, DEFAULT_CSV_IDS


def get_project_dir() -> Path:
    return Path(os.path.dirname(__file__)).parent.parent


@dataclass
class DurationDistribution:
    type: str = "NORMAL"
    mean: str = "NaN"  # Warning! this value is always interpreted as seconds
    arg1: str = "NaN"
    arg2: str = "NaN"
    unit: str = "seconds"  # This is the unit to show in the interface by transforming the values in seconds


def _should_consider_timer(delays: list) -> bool:
    """
    Default function to consider adding a timer when the percentage of delays observed for an activity being higher than 0s is more than
    the 10%.

    :param delays: list of delays observed for one activity (in seconds).
    :return: a boolean denoting if there should be a timer for that activity or not.
    """
    num_positive_delays = sum([delay > 0.0 for delay in delays])
    return (num_positive_delays / len(delays)) > 0.1


@dataclass
class Configuration:
    """Class storing the configuration parameters for the start time estimation.

    Attributes:
        log_ids                     Identifiers for each key element (e.g. executed activity or resource).
        num_evaluations             Number of iterations of the hyper-optimization search.
        num_evaluation_simulations  Number of simulations performed with each enhanced BPMN model to evaluate its quality.
        should_consider_timer       Function taking as input a list of seconds for all the delays that activity has registered, and
                                    returning a bool indicating if those delays should be considered as a timer, or discarded as outliers.
        process_name                Name of the process to use in the output files (BPMN and simulated log files).
    """
    log_ids: EventLogIDs = DEFAULT_CSV_IDS
    num_evaluations: int = 10
    num_evaluation_simulations: int = 5
    should_consider_timer: Callable[[list], bool] = _should_consider_timer
    process_name: str = "process_model"

    PATH_PROJECT = get_project_dir()
    PATH_INPUTS = PATH_PROJECT.joinpath("inputs")
    PATH_OUTPUTS = PATH_PROJECT.joinpath("outputs")
    PATH_EXTERNAL_TOOLS = PATH_PROJECT.joinpath("external_tools")
    PATH_BIMP = PATH_PROJECT.joinpath("external_tools").joinpath("simulator").joinpath("qbp-simulator-engine.jar")
