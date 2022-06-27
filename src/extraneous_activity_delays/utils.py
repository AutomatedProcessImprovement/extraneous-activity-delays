import datetime
import os
import shutil
import uuid
from pathlib import Path

import pandas as pd
from estimate_start_times.config import EventLogIDs


def split_log_training_validation_trace_wise(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        training_percentage: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the traces of [event_log] into two separated event logs (one for training and the other for validation). Split full traces in
    order to achieve an approximate proportion of [training_percentage] events in the training set.

    :param event_log:           event log to split.
    :param log_ids:             IDs for the columns of the event log.
    :param training_percentage: percentage of events (approx) to retain in the training data.

    :return: a tuple with two datasets, the training and the validation ones.
    """
    # Sort event log
    sorted_event_log = event_log.sort_values([log_ids.start_time, log_ids.end_time])
    # Take first trace until the number of events is [training_percentage] * total size
    total_events = len(event_log)
    training_case_ids = []
    training_full = False
    # Go over the case IDs (sorted by start and end time of its events)
    for case_id in sorted_event_log[log_ids.case].unique():
        # The first traces until the size limit is met goes to the training set
        if not training_full:
            training_case_ids += [case_id]
            training_full = len(event_log[event_log[log_ids.case].isin(training_case_ids)]) >= (training_percentage * total_events)
    # Return the two splits
    return (event_log[event_log[log_ids.case].isin(training_case_ids)],
            event_log[~event_log[log_ids.case].isin(training_case_ids)])


def split_log_training_validation_event_wise(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        training_percentage: float,
        remove_partial_traces_from_validation: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the traces of [event_log] into two separated event logs (one for training and the other for validation). Split event-wise retaining the
    first [training_percentage] of events in the training set, and the remaining ones in the validation set.

    :param event_log:                               event log to split.
    :param log_ids:                                 IDs for the columns of the event log.
    :param training_percentage:                     percentage of events to retain in the training data.
    :param remove_partial_traces_from_validation    if true, remove from validation set the traces that has been split being some event in
                                                    training and some events in validation.

    :return: a tuple with two datasets, the training and the validation ones.
    """
    # Sort event log
    sorted_event_log = event_log.sort_values([log_ids.end_time, log_ids.start_time])
    # Get the two splits
    num_train_events = int(len(event_log) * training_percentage)
    training_log = sorted_event_log.head(num_train_events)
    num_validation_events = len(event_log) - num_train_events
    validation_log = sorted_event_log.tail(num_validation_events)
    # Remove from validation incomplete traces
    if remove_partial_traces_from_validation:
        training_cases = training_log[log_ids.case].unique()
        validation_log = validation_log[~validation_log[log_ids.case].isin(training_cases)]
    # Return the two splits
    return training_log, validation_log


def delete_folder(folder_path: str):
    shutil.rmtree(folder_path, ignore_errors=True)


def create_new_tmp_folder(base_path: Path) -> Path:
    # Get non existent folder name
    output_folder = base_path.joinpath(datetime.datetime.today().strftime('%Y%m%d_') + str(uuid.uuid4()).upper().replace('-', '_'))
    while not create_folder(output_folder):
        output_folder = base_path.joinpath(datetime.datetime.today().strftime('%Y%m%d_') + str(uuid.uuid4()).upper().replace('-', '_'))
    # Return P  ath to new folder
    return output_folder


def create_folder(path: Path) -> bool:
    if os.path.exists(path):
        return False
    else:
        os.makedirs(path)
        return True
