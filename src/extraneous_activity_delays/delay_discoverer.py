import pandas as pd
from estimate_start_times.config import Configuration as StartTimeConfiguration, ConcurrencyOracleType, ReEstimationMethod, \
    ResourceAvailabilityType
from estimate_start_times.config import EventLogIDs as StartTimeEventLogIDs
from estimate_start_times.estimator import StartTimeEstimator

from extraneous_activity_delays.config import EventLogIDs
from extraneous_activity_delays.infer_distribution import infer_distribution


def calculate_extraneous_activity_delays(
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
    denoting if a timer should be considered or not. By default, always consider a timer unless there is some delay different from 0.
    :return: a dictionary with the activity name as key and the time distribution of its delay.
    """
    # Calculate estimated start times (with enablement and resource availability)
    log_ids = config.log_ids
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
        delays = [delay.total_seconds() if delay > datetime.timedelta(0) else 0.0 for delay in delays]
        if should_consider_timer(delays):
            timers[activity] = infer_distribution(delays)
    # Return the delays
    return timers
