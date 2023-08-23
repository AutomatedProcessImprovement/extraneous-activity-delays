import json

import pandas as pd
from extraneous_activity_delays.config import Configuration, TimerPlacement
from extraneous_activity_delays.delay_discoverer import (
    compute_complex_extraneous_activity_delays,
    compute_naive_extraneous_activity_delays,
)
from extraneous_activity_delays.utils.file_manager import create_folder
from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import RCalendar
from pix_framework.io.event_log import EventLogIDs, read_csv_log

log_ids = EventLogIDs(
    case="case_id", activity="activity", resource="resource", start_time="start_time", end_time="end_time"
)
processes = [
    "Insurance_Claims",
    "Insurance_Claims_1_timer",
    "Insurance_Claims_3_timers",
    "Insurance_Claims_5_timers",
    "Loan_Application",
    "Loan_Application_1_timer",
    "Loan_Application_3_timers",
    "Loan_Application_5_timers",
    "Pharmacy",
    "Pharmacy_1_timer",
    "Pharmacy_3_timers",
    "Pharmacy_5_timers",
    "Procure_to_Pay",
    "Procure_to_Pay_1_timer",
    "Procure_to_Pay_3_timers",
    "Procure_to_Pay_5_timers",
]


def inf_sys_evaluation():
    # --- Evaluation folder --- #
    evaluation_folder = Configuration().PATH_OUTPUTS.joinpath("synthetic-evaluation").joinpath("individual")
    create_folder(evaluation_folder)
    smape_file_path = evaluation_folder.joinpath("smape.csv")
    with open(smape_file_path, "a") as file:
        file.write("dataset,naive_sMAPE,complex_sMAPE,complex_adj_sMAPE,naive_MAPE,complex_MAPE,complex_adj_MAPE\n")
    for process in processes:
        # --- Paths --- #
        synthetic_input_path = Configuration().PATH_INPUTS.joinpath("synthetic")
        simulation_params_path = str(synthetic_input_path.joinpath(process + ".json"))
        log_path = str(synthetic_input_path.joinpath(process + ".csv.gz"))
        naive_log_path = str(evaluation_folder.joinpath(process + "_naive_enhanced.csv.gz"))
        complex_log_path = str(evaluation_folder.joinpath(process + "_complex_enhanced.csv.gz"))
        complex_adj_log_path = str(evaluation_folder.joinpath(process + "_complex_adj_enhanced.csv.gz"))
        # --- Read event log --- #
        event_log = read_csv_log(log_path, log_ids)
        # --- Read simulation model --- #
        with open(simulation_params_path) as json_file:
            simulation_parameters = json.load(json_file)
        working_schedules = _json_schedules_to_rcalendar(simulation_parameters)
        # --- Configurations --- #
        configuration = Configuration(
            log_ids=log_ids,
            process_name=process,
            timer_placement=TimerPlacement.BEFORE,
            working_schedules=working_schedules,
        )
        configuration_adjusted = Configuration(
            log_ids=log_ids,
            process_name=process,
            timer_placement=TimerPlacement.BEFORE,
            working_schedules=working_schedules,
            extrapolate_complex_delays_estimation=True,
        )
        # --- Discover individual extraneous delays --- #
        naive_enhanced_event_log = compute_naive_extraneous_activity_delays(
            event_log, configuration, configuration.should_consider_timer, experimentation=True
        )
        naive_enhanced_event_log.to_csv(naive_log_path, index=False)
        complex_enhanced_event_log = compute_complex_extraneous_activity_delays(
            event_log, configuration, configuration.should_consider_timer, experimentation=True
        )
        complex_enhanced_event_log.to_csv(complex_log_path, index=False)
        complex_adj_enhanced_event_log = compute_complex_extraneous_activity_delays(
            event_log, configuration_adjusted, configuration.should_consider_timer, experimentation=True
        )
        complex_adj_enhanced_event_log.to_csv(complex_adj_log_path, index=False)
        # --- Measure error --- #
        smape_naive = _compute_smape(naive_enhanced_event_log)
        smape_complex = _compute_smape(complex_enhanced_event_log)
        smape_complex_adj = _compute_smape(complex_adj_enhanced_event_log)
        mape_naive = _compute_mape(naive_enhanced_event_log)
        mape_complex = _compute_mape(complex_enhanced_event_log)
        mape_complex_adj = _compute_mape(complex_adj_enhanced_event_log)
        with open(smape_file_path, "a") as file:
            file.write(
                "{},{},{},{},{},{},{}\n".format(
                    process, smape_naive, smape_complex, smape_complex_adj, mape_naive, mape_complex, mape_complex_adj
                )
            )


def _compute_smape(event_log: pd.DataFrame) -> float:
    # Get activity instances with either estimated delay or actual delay
    estimated = event_log[(event_log["estimated_extraneous_delay"] > 0.0) | (event_log["extraneous_delay"] > 0.0)]
    # Compute smape
    if len(estimated) > 0:
        smape = sum(
            [
                2
                * abs(delays["estimated_extraneous_delay"] - delays["extraneous_delay"])
                / (delays["extraneous_delay"] + delays["estimated_extraneous_delay"])
                for index, delays in estimated[["estimated_extraneous_delay", "extraneous_delay"]].iterrows()
            ]
        ) / len(estimated)
    else:
        smape = 0.0
    # Return value
    return smape


def _compute_mape(event_log: pd.DataFrame) -> float:
    # Get activity instances with either estimated delay or actual delay
    estimated = event_log[(event_log["estimated_extraneous_delay"] > 0.0) | (event_log["extraneous_delay"] > 0.0)]
    # Compute mape
    if len(estimated) > 0:
        mape = sum(
            [
                abs((delays["extraneous_delay"] - delays["estimated_extraneous_delay"]) / delays["extraneous_delay"])
                for index, delays in estimated[["estimated_extraneous_delay", "extraneous_delay"]].iterrows()
            ]
        ) / len(estimated)
    else:
        mape = 0.0
    # Return value
    return mape


def _json_schedules_to_rcalendar(simulation_parameters: dict) -> dict:
    """
    Transform the calendars specified as part of the simulation parameters to a dict with the ID of the resources as key, and their
    calendar (RCalendar) as value.

    :param simulation_parameters: dictionary with the parameters for prosimos simulation.

    :return: a dict with the ID of the resources as key and their calendar as value.
    """
    # Read calendars
    calendars = {}
    for calendar in simulation_parameters["resource_calendars"]:
        r_calendar = RCalendar(calendar["id"])
        for slot in calendar["time_periods"]:
            r_calendar.add_calendar_item(slot["from"], slot["to"], slot["beginTime"], slot["endTime"])
        calendars[r_calendar.calendar_id] = r_calendar
    # Assign calendars to each resource
    resource_calendars = {}
    for profile in simulation_parameters["resource_profiles"]:
        for resource in profile["resource_list"]:
            if int(resource["amount"]) > 1:
                for i in range(int(resource["amount"])):
                    resource_calendars["{}_{}".format(resource["name"], i)] = calendars[resource["calendar"]]
            else:
                resource_calendars[resource["name"]] = calendars[resource["calendar"]]
    # Return resource calendars
    return resource_calendars


if __name__ == "__main__":
    inf_sys_evaluation()
