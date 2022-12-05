from typing import Callable

import pandas as pd

from estimate_start_times.config import Configuration as StartTimeConfiguration, ConcurrencyOracleType, ReEstimationMethod, \
    ResourceAvailabilityType
from estimate_start_times.estimator import StartTimeEstimator
from extraneous_activity_delays.config import Configuration, SimulationEngine
from extraneous_activity_delays.prosimos.infer_distribution import infer_distribution as infer_distribution_prosimos
from extraneous_activity_delays.qbp.infer_distribution import infer_distribution as infer_distribution_qbp


def compute_extraneous_activity_delays(
        event_log: pd.DataFrame,
        config: Configuration,
        should_consider_timer: Callable[[list], bool] = lambda delays: sum(delays) > 0.0
) -> dict:
    """
    Calculate, for each activity, the distribution of its extraneous delays. I.e., the distribution of the time passed since the
    activity is both enabled and its resource available, and the recorded start of the activity.

    :param event_log: Event log storing the information of the process.
    :param config: configuration of the estimation search.
    :param should_consider_timer: lambda function that, given a list of floats representing all the delays registered, returns a boolean
    denoting if a timer should be considered or not. By default, no consider timer if all delays are 0.
    :return: a dictionary with the activity name as key and the time distribution of its delay.
    """
    # Calculate estimated start times (with enablement and resource availability)
    log_ids = config.log_ids
    start_time_config = StartTimeConfiguration(
        log_ids=log_ids,
        concurrency_oracle_type=ConcurrencyOracleType.HEURISTICS,
        re_estimation_method=ReEstimationMethod.MODE,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
        bot_resources=config.bot_resources,
        instant_activities=config.instant_activities,
        consider_start_times=True
    )
    enhanced_event_log = StartTimeEstimator(event_log, start_time_config).estimate()
    # Discover the time distribution of each activity's delay
    timers = {}
    for activity in enhanced_event_log[log_ids.activity].unique():
        # Get the activity instances which start was estimated with, at least, the enabled time
        activity_instances = enhanced_event_log[
            (enhanced_event_log[log_ids.activity] == activity) &  # An execution of this activity, AND
            (~pd.isna(enhanced_event_log[log_ids.estimated_start_time]))  # having an enabled time
            ]
        # Transform the delay to seconds
        delays = [
            delay.total_seconds()
            for delay in (activity_instances[log_ids.start_time] - activity_instances[log_ids.estimated_start_time])
        ]
        # If the delay should be considered, add it
        if should_consider_timer(delays):
            if config.simulation_engine == SimulationEngine.PROSIMOS:
                timers[activity] = infer_distribution_prosimos(delays)
            else:
                timers[activity] = infer_distribution_qbp(delays)
    # Return the delays
    return timers
