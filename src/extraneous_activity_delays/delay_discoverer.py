import pandas as pd
from estimate_start_times.config import Configuration as StartTimeConfiguration, ConcurrencyOracleType, ReEstimationMethod, \
    ResourceAvailabilityType
from estimate_start_times.config import EventLogIDs as StartTimeEventLogIDs
from estimate_start_times.estimator import StartTimeEstimator
from numpy import mean

from extraneous_activity_delays.config import EventLogIDs, DurationDistribution
from extraneous_activity_delays.temp_infer_distribution import best_fit_distribution


def calculate_extraneous_activity_delays(event_log: pd.DataFrame, log_ids: EventLogIDs) -> dict:
    """
    Calculate, for each activity, the distribution of its extraneous delays. I.e., the distribution of the time passed since the
    activity is both enabled and its resource available, and the recorded start of the activity.

    :param event_log: Event log storing the information of the process.
    :param log_ids: id mapping for each of the columns.
    :return: a dictionary with the activity name as key and the time distribution of its delay.
    """
    # Calculate estimated start times (with enablement and resource availability)
    start_time_config = StartTimeConfiguration(
        log_ids=StartTimeEventLogIDs(
            case=log_ids.case,
            activity=log_ids.activity,
            start_time=log_ids.start_time,
            end_time=log_ids.end_time,
            enabled_time=log_ids.enabled_time,
            available_time=log_ids.available_time,
            estimated_start_time=log_ids.estimated_start_time,
            resource=log_ids.resource
        ),
        concurrency_oracle_type=ConcurrencyOracleType.HEURISTICS,
        re_estimation_method=ReEstimationMethod.MODE,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
        consider_start_times=True
    )
    enhanced_event_log = StartTimeEstimator(event_log, start_time_config).estimate()
    # Discover the time distribution of each activity's delay
    timers = {}
    for activity in enhanced_event_log[log_ids.activity].unique():
        delays = (enhanced_event_log[enhanced_event_log[log_ids.activity] == activity][log_ids.start_time] -
                  enhanced_event_log[enhanced_event_log[log_ids.activity] == activity][log_ids.estimated_start_time])
        # TODO estimate a distribution and its parameters
        timers[activity] = DurationDistribution(
            type="NORMAL",
            mean=str(int(mean(delays).total_seconds()))
        )
        # Return the delays
    return timers
