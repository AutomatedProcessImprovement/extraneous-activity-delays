from typing import Callable, Tuple

import pandas as pd
from pix_framework.calendar.availability import absolute_unavailability_intervals_within
from pix_framework.log_ids import EventLogIDs
from pix_framework.statistics.distribution import get_best_fitting_distribution
from start_time_estimator.concurrency_oracle import OverlappingConcurrencyOracle
from start_time_estimator.config import Configuration as StartTimeConfiguration
from start_time_estimator.resource_availability import CalendarResourceAvailability

from extraneous_activity_delays.config import Configuration, TimerPlacement


def compute_naive_extraneous_activity_delays(
    event_log: pd.DataFrame,
    config: Configuration,
    should_consider_timer: Callable[[list], bool] = lambda delays: sum(delays) > 0.0,
) -> dict:
    """
    Compute, for each activity, the distribution of its extraneous delays. I.e., the distribution of the time passed since the
    activity is both enabled and its resource available, and the recorded start of the activity.

    :param event_log:               Event log storing the information of the process.
    :param config:                  Configuration of the estimation search.
    :param should_consider_timer:   Lambda function that, given a list of floats representing all the delays registered, returns a boolean
                                    denoting if a timer should be considered or not. By default, no consider timer if all delays are 0.

    :return: a dictionary with the activity name as key and the time distribution of its delay.
    """
    log_ids = config.log_ids
    # Compute both enablement and resource availability times
    start_time_config = StartTimeConfiguration(
        log_ids=log_ids,
        concurrency_thresholds=config.concurrency_thresholds,
        working_schedules=config.working_schedules,
    )
    if _should_compute_enabled_times(event_log, config):
        concurrency_oracle = OverlappingConcurrencyOracle(event_log, start_time_config)
        concurrency_oracle.add_enabled_times(event_log, set_nat_to_first_event=True, include_enabling_activity=True)
    if log_ids.available_time not in event_log.columns:
        resource_availability = CalendarResourceAvailability(event_log, start_time_config)
        resource_availability.add_resource_availability_times(event_log)
    # Who to impute the extraneous delay to: the executed activity if the timer goes before, the enabling activity if it goes after
    impute_to = log_ids.activity if config.timer_placement == TimerPlacement.BEFORE else log_ids.enabling_activity
    # Discover the time distribution of each activity's delay
    timers = {}
    for activity, instances in event_log.groupby(impute_to):
        # Get the activity instances with enabled time
        filtered_instances = instances[(~pd.isna(instances[log_ids.enabled_time]))]
        # Compute the extraneous delays in seconds
        delays = [
            delay.total_seconds()
            for delay in filtered_instances[log_ids.start_time]
            - filtered_instances[[log_ids.enabled_time, log_ids.available_time]].max(
                axis=1, skipna=True, numeric_only=False
            )
        ]
        # If the delay should be considered, add it
        if should_consider_timer(delays):
            timers[activity] = get_best_fitting_distribution(delays)
    # Return the delays
    return timers


def compute_complex_extraneous_activity_delays(
    event_log: pd.DataFrame,
    config: Configuration,
    should_consider_timer: Callable[[list], bool] = lambda delays: sum(delays) > 0.0,
) -> dict:
    """
    Compute, for each activity, the distribution of its extraneous delays. To compute the extraneous delay of an activity instance,
    detect the first and lasts instants in time in which the activity was enabled and the resource available for processing it (taking
    into account both the resource contention and availability calendars). The extraneous delay is the interval between these two
    instants, no matter if the resource became unavailable in the middle.

    :param event_log:               Event log storing the information of the process.
    :param config:                  Configuration of the estimation search.
    :param should_consider_timer:   Lambda function that, given a list of floats representing all the delays registered, returns a boolean
                                    denoting if a timer should be considered or not. By default, no consider timer if all delays are 0.

    :return: a dictionary with the activity name as key and the time distribution of its delay.
    """
    # Compute enabled time of each activity instance
    log_ids = config.log_ids
    start_time_config = StartTimeConfiguration(
        log_ids=log_ids,
        concurrency_thresholds=config.concurrency_thresholds,
        working_schedules=config.working_schedules,
    )
    if _should_compute_enabled_times(event_log, config):
        concurrency_oracle = OverlappingConcurrencyOracle(event_log, start_time_config)
        concurrency_oracle.add_enabled_times(event_log, set_nat_to_first_event=True, include_enabling_activity=True)
    # Compute first and last instants where the resource was available
    _extend_log_with_first_last_available(event_log, log_ids, config)
    # Who to impute the extraneous delay to: the executed activity if the timer goes before, the enabling activity if it goes after
    impute_to = log_ids.activity if config.timer_placement == TimerPlacement.BEFORE else log_ids.enabling_activity
    # Discover the time distribution of each activity's delay
    timers = {}
    for activity, instances in event_log.groupby(impute_to):
        # Get the activity instances with enabled time
        filtered_instances = instances[(~pd.isna(instances[log_ids.enabled_time]))]
        # Transform the delay to seconds
        delays = [
            delay.total_seconds()
            for delay in (filtered_instances["last_available"] - filtered_instances["first_available"])
            if not pd.isna(delay)
        ]
        # If the delay should be considered, add it
        if should_consider_timer(delays):
            timers[activity] = get_best_fitting_distribution(delays)
    # Remove extra columns
    event_log.drop(["last_available", "first_available"], axis=1, inplace=True)
    # Return discovered delays
    return timers


def _extend_log_with_first_last_available(event_log: pd.DataFrame, log_ids: EventLogIDs, config: Configuration):
    """
    Add, to [event_log], two columns with the first and last timestamps in which the resource that performed that activity was available.

    :param event_log:   Event log storing the information of the process.
    :param log_ids:     Mapping for the columns in the event log.
    :param config:      Configuration of the estimation search.
    """
    # Initiate both first and last available columns to NaT
    event_log["first_available"] = pd.NaT
    event_log["last_available"] = pd.NaT
    for resource, events in event_log.groupby(log_ids.resource):
        # Initialize resource working calendar if existing
        calendar = config.working_schedules[resource] if resource in config.working_schedules else None
        indexes, first_available, last_available = [], [], []
        for index, event in events.iterrows():
            # Get activity instances performed by the same resource happening in its waiting time
            performed_events = events[
                (
                    (event[log_ids.enabled_time] < events[log_ids.end_time])
                    & (events[log_ids.end_time] <= event[log_ids.start_time])
                )
                | (
                    (event[log_ids.enabled_time] <= events[log_ids.start_time])
                    & (events[log_ids.start_time] < event[log_ids.start_time])
                )
            ]
            # If the resource has a calendar associated, get off-duty intervals happening in its waiting time
            if calendar:
                resource_off_duty = absolute_unavailability_intervals_within(
                    start=event[log_ids.enabled_time],
                    end=event[log_ids.start_time],
                    schedule=calendar,
                )
            else:
                resource_off_duty = []
            # Get first and last availability instants
            indexes += [index]
            first_instant, last_instant = _get_first_and_last_available(
                beginning=event[log_ids.enabled_time],
                end=event[log_ids.start_time],
                starts=list(performed_events[log_ids.start_time]) + [interval.start for interval in resource_off_duty],
                ends=list(performed_events[log_ids.end_time]) + [interval.end for interval in resource_off_duty],
                time_gap=config.time_gap,
                extrapolate=config.extrapolate_complex_delays_estimation,
            )
            if first_instant:
                # Available instants found
                first_available += [first_instant]
                last_available += [last_instant]
            else:
                # Busy during all the waiting time, set start time as availability
                first_available += [event[log_ids.start_time]]
                last_available += [event[log_ids.start_time]]
        # Set first and last available times for all events of this resource
        event_log.loc[indexes, "first_available"] = first_available
        event_log.loc[indexes, "last_available"] = last_available
    # Convert columns to Timestamp
    event_log["first_available"] = pd.to_datetime(event_log["first_available"], utc=True)
    event_log["last_available"] = pd.to_datetime(event_log["last_available"], utc=True)


def _get_first_and_last_available(
    beginning: pd.Timestamp,
    end: pd.Timestamp,
    starts: list,
    ends: list,
    time_gap: pd.Timedelta,
    extrapolate: bool = False,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get the first instant from the period [from]-[to] where the resource was available for a [time_gap] amount of time.

    :param beginning:   Start of the interval where to search.
    :param end:         End of the interval where to search.
    :param starts:      List of instants where either a non-working period or an activity instance started.
    :param ends:        List of instants where either an activity instance or a non-working period finished.
    :param time_gap:    Minimum time gap required for a non-working period to be considered as such.
    :param extrapolate: If 'True', instead of getting the first available time as such, move it to half its distance
                        between itself and the beginning of the interval. For example, if the beginning is at 1, and
                        the discovered first available time is at 5, the extrapolated one is 3 (the middle point). The
                        same is done with the last available and the end of the interval. The objective is to reduce
                        potential mis-estimations as the real first available time is unknown.

    :return: A tuple with the first and last timestamps within all [start] and [end] timestamps where the
    resource was available for a [time_gap] amount of time.
    """
    # Add beginning and end of interval as artificial instant activities
    starts += [beginning, end]
    ends += [beginning, end]
    # Store the start and ends in a list of dicts
    times = (
        pd.DataFrame(
            {
                "time": starts + ends,
                "type": ["start"] * len(starts) + ["end"] * len(ends),
            }
        )
        .sort_values(["time", "type"], ascending=[True, False])
        .values.tolist()
    )
    first_available = None
    last_available = None
    # Go over them start->end, until a moment with no active unavailable intervals is reached
    i = 0
    active = 0  # Number of active unavailable intervals
    while not first_available and i < len(times):
        # Increase active unavailable intervals if current timestamps is 'start', or decrease otherwise
        active += 1 if times[i][1] == "start" else -1
        # Check if no active unavailable intervals
        if active == 0 and (  # No active unavailable intervals at this point, and
            i + 1 == len(times) or times[i + 1][0] - times[i][0] >= time_gap  # either this is the last point, or
        ):  # there is an available time gap with enough duration
            # Resource available at this point, check time gap until next event
            first_available = times[i][0]
        i += 1
    # If time gap found, search for last available, not necessary otherwise
    if first_available:
        # Go over them end->start, until a moment with no active unavailable intervals is reached
        i = len(times) - 1  # Index to go over the timestamps
        active = 0  # Number of active unavailable intervals
        while not last_available and i > 0:
            if times[i][0] <= first_available:
                # If the search reached [first_available], set to it and finish
                last_available = first_available
            else:
                # Increase active unavailable intervals if current timestamps is 'end', or decrease otherwise
                active += 1 if times[i][1] == "end" else -1
                # Check if no active unavailable intervals
                if active == 0 and (  # No active unavailable intervals at this point, and
                    i == 0
                    or times[i][0] - times[i - 1][0] >= time_gap  # either this is the last point in the search, or
                ):  # there is an available time gap with enough duration
                    # Resource available at this point, check time gap until next event
                    last_available = times[i][0]
            i -= 1
    # Return first available
    if extrapolate:
        first_available = first_available - ((first_available - beginning) / 2)
        last_available = last_available + ((end - last_available) / 2)
    return first_available, last_available


def _should_compute_enabled_times(event_log: pd.DataFrame, config: Configuration):
    return config.log_ids.enabled_time not in event_log.columns or (
        config.timer_placement == TimerPlacement.AFTER and config.log_ids.enabling_activity not in event_log.columns
    )
