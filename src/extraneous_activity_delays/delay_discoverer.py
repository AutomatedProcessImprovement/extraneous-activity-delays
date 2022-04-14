import pandas as pd
from estimate_start_times.config import Configuration as StartTimeConfiguration, ConcurrencyOracleType, ReEstimationMethod, \
    ResourceAvailabilityType
from estimate_start_times.config import EventLogIDs as StartTimeEventLogIDs
from estimate_start_times.estimator import StartTimeEstimator

from waiting_times.config import EventLogIDs


def add_waiting_times(event_log: pd.DataFrame, log_ids: EventLogIDs):
    # Calculate estimated start times (with enablement and resource availability)
    start_time_config = StartTimeConfiguration(
        log_ids=StartTimeEventLogIDs(
            case=log_ids.case,
            activity=log_ids.activity,
            start_time=log_ids.estimated_start_time,  # Assign other ID to not override the start time
            end_time=log_ids.end_time,
            enabled_time=log_ids.enabled_time,
            available_time=log_ids.available_time,
            resource=log_ids.resource
        ),
        concurrency_oracle_type=ConcurrencyOracleType.HEURISTICS,
        re_estimation_method=ReEstimationMethod.MODE,
        resource_availability_type=ResourceAvailabilityType.SIMPLE
    )
    enhanced_event_log = StartTimeEstimator(event_log, start_time_config).estimate()
    # Associate the time between the estimated start time and the real start time as WT
