import datetime
import math

import pandas as pd
from estimate_start_times.config import EventLogIDs
from scipy.stats import wasserstein_distance


def discretize_to_hour(seconds: int):
    return math.floor(seconds / 3600)


def discretize_to_day(seconds: int):
    return math.floor(seconds / 3600 / 24)


def absolute_hour_emd(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        discretize=discretize_to_hour  # function to discretize a total amount of seconds
) -> float:
    # Get the first and last dates of the log
    interval_start = min(event_log_1[log_1_ids.start_time].min(), event_log_2[log_2_ids.start_time].min()).floor(freq='H')
    # Discretize each instant to its corresponding "bin"
    discretized_instants_1 = [
        discretize(difference.total_seconds()) for difference in (event_log_1[log_1_ids.start_time] - interval_start)
    ]
    discretized_instants_1 += [
        discretize(difference.total_seconds()) for difference in (event_log_1[log_1_ids.end_time] - interval_start)
    ]
    # Discretize each instant to its corresponding "bin"
    discretized_instants_2 = [
        discretize(difference.total_seconds()) for difference in (event_log_2[log_2_ids.start_time] - interval_start)
    ]
    discretized_instants_2 += [
        discretize(difference.total_seconds()) for difference in (event_log_2[log_2_ids.end_time] - interval_start)
    ]
    # Return EMD metric
    return wasserstein_distance(discretized_instants_1, discretized_instants_2)


def trace_duration_emd(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        bin_size: datetime.timedelta
) -> float:
    # Get trace durations of each trace for the first log
    trace_durations_1 = []
    for case, events in event_log_1.groupby([log_1_ids.case]):
        trace_durations_1 += [events[log_1_ids.end_time].max() - events[log_1_ids.start_time].min()]
    # Get trace durations of each trace for the second log
    trace_durations_2 = []
    for case, events in event_log_2.groupby([log_2_ids.case]):
        trace_durations_2 += [events[log_2_ids.end_time].max() - events[log_2_ids.start_time].min()]
    # Discretize each instant to its corresponding "bin"
    min_duration = min(trace_durations_1 + trace_durations_2)
    discretized_durations_1 = [math.floor((trace_duration - min_duration) / bin_size) for trace_duration in trace_durations_1]
    discretized_durations_2 = [math.floor((trace_duration - min_duration) / bin_size) for trace_duration in trace_durations_2]
    # Return EMD metric
    return wasserstein_distance(discretized_durations_1, discretized_durations_2)
