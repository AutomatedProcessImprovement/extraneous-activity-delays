import enum
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
from lxml.etree import ElementTree

from pix_utils.log_ids import EventLogIDs, DEFAULT_CSV_IDS


def get_project_dir() -> Path:
    return Path(os.path.dirname(__file__)).parent.parent


@dataclass
class SimulationModel:
    bpmn_document: ElementTree
    simulation_parameters: dict = None


class SimulationEngine(enum.Enum):
    PROSIMOS = 1
    QBP = 2


class SimulationOutput(enum.Enum):
    SUCCESS = 1
    ERROR = 2


class OptimizationMetric(enum.Enum):
    CYCLE_TIME = 1
    ABSOLUTE_EMD = 2
    CIRCADIAN_EMD = 3
    RELATIVE_EMD = 4


class TimerPlacement(enum.Enum):
    BEFORE = 1
    AFTER = 2


def _should_consider_timer(delays: list) -> bool:
    """
    Default function to consider adding a timer when the percentage of delays observed for an activity being higher than 0s is more than
    the 10%.

    :param delays: list of delays observed for one activity (in seconds).
    :return: a boolean denoting if there should be a timer for that activity or not.
    """
    num_positive_delays = sum([delay > 0.0 for delay in delays])
    return len(delays) > 0 and (num_positive_delays / len(delays)) > 0.05


@dataclass
class Configuration:
    """
    Class storing the configuration parameters for the extraneous activity delay optimizer.

    Extraneous delays options:
        timer_placement             Option to consider the placement of the timers either BEFORE (the extraneous delay is considered to be
                                    happening previously to an activity instance) or AFTER (the extraneous delay is considered to be
                                    happening afterward an activity instance) each activity.
        simulation_engine           Simulation engine to use during the optimization process (e.g. Prosimos).
        optimization_metric         Metric to optimize during the optimization process.

    Extraneous delays parameters:
        should_consider_timer       Function taking as input a list of seconds for all the delays that activity has registered, and
                                    returning a bool indicating if those delays should be considered as a timer, or discarded as outliers.
        working_schedules           Dictionary with the resources as key and the working calendars (RCalendar) as value.
        time_gap                    For complex extraneous delays, maximum interval of time allowed from a work item (task of off-duty
                                    interval) to another work item to not consider that the resource was available in that interval. For
                                    example, a resource works in a task from 16.00 to 17.49, but their off-duty period starts at 18.49. If
                                    time-gap is set to >11 minutes, it won't be counted as an available period.

    Optimization process parameters:
        num_iterations              Number of iterations of the hyper-optimization search.
        num_evaluation_simulations  Number of simulations performed with each enhanced BPMN model to evaluate its quality.
        max_alpha                   Maximum scale factor to multiply the discovered timers in the hyper-optimization.
        training_partition_ratio    Float value between 0.0 and 1.0 (or None). If None, train and validate the hyper-parametrization using
                                    the full event log passed as argument. Otherwise, perform the hyper-parametrization with a hold-out
                                    retaining this percentage of events (in full traces) for the training set. For example, if the value is
                                    0.8 it will train with approximately a subset of traces of the event log with the 80% of the total
                                    events, and validate with the remaining traces.

    Enabled time estimation:
        bot_resources               Set of resource IDs corresponding bots, in order to set the estimated start time of its events as
                                    their end time.
        instant_activities          Set of instantaneous activities, in order to set their estimated start time as their end time.

    General parameters:
        process_name                Name of the process to use in the output files (BPMN and simulated log files).
        log_ids                     Identifiers for each key element (e.g. executed activity or resource).
        debug                       Boolean denoting whether to print debug information or not.
    """
    # Extraneous delays options
    timer_placement: TimerPlacement = TimerPlacement.BEFORE
    simulation_engine: SimulationEngine = SimulationEngine.PROSIMOS
    optimization_metric: OptimizationMetric = OptimizationMetric.RELATIVE_EMD
    # Extraneous delays parameters
    should_consider_timer: Callable[[list], bool] = _should_consider_timer
    working_schedules: dict = field(default_factory=dict)
    time_gap: pd.Timedelta = pd.Timedelta(0)
    # Optimization process parameters
    num_iterations: int = 100
    num_evaluation_simulations: int = 10
    max_alpha: float = 1.0
    training_partition_ratio: float = None
    # Enabled time estimation
    bot_resources: set = field(default_factory=set)
    instant_activities: set = field(default_factory=set)
    # General parameters
    process_name: str = "process"
    log_ids: EventLogIDs = field(default_factory=lambda: DEFAULT_CSV_IDS)
    debug: bool = False
    # Paths
    PATH_PROJECT = get_project_dir()
    PATH_INPUTS = PATH_PROJECT.joinpath("inputs")
    PATH_OUTPUTS = PATH_PROJECT.joinpath("outputs")
    PATH_EXTERNAL_TOOLS = PATH_PROJECT.joinpath("external_tools")
    PATH_QBP = PATH_PROJECT.joinpath("external_tools").joinpath("simulator").joinpath("qbp-simulator-engine.jar")
