from dataclasses import dataclass

from estimate_start_times.config import EventLogIDs, DEFAULT_CSV_IDS


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
        concurrency_oracle_type     Concurrency oracle to use (e.g. heuristics miner's concurrency oracle).
        resource_availability_type  Resource availability engine to use (e.g. using resource calendars).
        missing_resource            String to identify the events with missing resource (it is avoided in the resource availability
                                    calculation).
        non_estimated_time          Time to use as value when the start time cannot be estimated (later re-estimated with
                                    [re_estimation_method].
        re_estimation_method        Method (e.g. median) to re-estimate the start times that couldn't be estimated due to lack of resource
                                    availability and causal predecessors.
        bot_resources               Set of resource IDs corresponding bots, in order to set their events as instant.
        instant_activities          Set of instantaneous activities, in order to set their events as instant.
        heuristics_thresholds       Thresholds for the heuristics concurrency oracle (only used is this oracle is selected as
                                    [concurrency_oracle_type].
        consider_parallelism        Consider start times when checking for the enabled time of an activity in the concurrency oracle, if
                                    'true', do not consider the events which end time is after the start time of the current activity
                                    instance, they overlap so no causality between them.
        outlier_statistic           Statistic (e.g. median) to calculate the most typical duration from the distribution of each activity
                                    durations to consider and re-estimate the outlier events which estimated duration is higher.
        outlier_threshold           Threshold to control outliers, those events with estimated durations over
    """
    log_ids: EventLogIDs = DEFAULT_CSV_IDS
