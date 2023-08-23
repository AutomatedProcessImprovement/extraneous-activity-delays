import json
from pathlib import Path

from extraneous_activity_delays.config import (
    Configuration,
    DiscoveryMethod,
    SimulationEngine,
    SimulationModel,
    TimerPlacement,
)
from extraneous_activity_delays.enhance_with_delays import DirectEnhancer
from extraneous_activity_delays.utils.file_manager import create_folder
from lxml import etree
from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import RCalendar
from pix_framework.io.event_log import EventLogIDs, read_csv_log

log_ids = EventLogIDs(
    case="case_id", activity="activity", resource="resource", start_time="start_time", end_time="end_time"
)
processes = [
    ("Confidential_noTimers", "Confidential"),
    ("Insurance_Claims", "Insurance_Claims_5_timers_after"),
    ("Loan_Application", "Loan_Application_5_timers_after"),
    ("Pharmacy", "Pharmacy_5_timers_after"),
    ("Procure_to_Pay", "Procure_to_Pay_5_timers_after"),
]


def inf_sys_evaluation():
    # Run for each process
    for no_timers_process, process in processes:
        # --- Raw paths --- #
        real_input_path = Configuration().PATH_INPUTS.joinpath("synthetic")
        simulation_bpmn_path = str(real_input_path.joinpath(no_timers_process + ".bpmn"))
        simulation_params_path = str(real_input_path.joinpath(no_timers_process + ".json"))
        log_path = str(real_input_path.joinpath(process + ".csv.gz"))

        # --- Evaluation folder --- #
        evaluation_folder = (
            Configuration().PATH_OUTPUTS.joinpath("synthetic-evaluation").joinpath("before-after").joinpath(process)
        )
        create_folder(evaluation_folder)

        # --- Read event logs --- #
        event_log = read_csv_log(log_path, log_ids)

        # --- Read simulation models --- #
        parser = etree.XMLParser(remove_blank_text=True)
        simulation_bpmn_model = etree.parse(simulation_bpmn_path, parser)
        with open(simulation_params_path) as json_file:
            simulation_parameters = json.load(json_file)
        simulation_model = SimulationModel(simulation_bpmn_model, simulation_parameters)
        working_schedules = _json_schedules_to_rcalendar(simulation_parameters)

        # --- Configurations --- #
        config_naive_before = Configuration(
            log_ids=log_ids,
            process_name=process,
            discovery_method=DiscoveryMethod.NAIVE,
            timer_placement=TimerPlacement.BEFORE,
            simulation_engine=SimulationEngine.PROSIMOS,
            working_schedules=working_schedules,
        )
        config_complex_before = Configuration(
            log_ids=log_ids,
            process_name=process,
            discovery_method=DiscoveryMethod.COMPLEX,
            timer_placement=TimerPlacement.BEFORE,
            simulation_engine=SimulationEngine.PROSIMOS,
            working_schedules=working_schedules,
        )
        config_naive_after = Configuration(
            log_ids=log_ids,
            process_name=process,
            discovery_method=DiscoveryMethod.NAIVE,
            timer_placement=TimerPlacement.AFTER,
            simulation_engine=SimulationEngine.PROSIMOS,
            working_schedules=working_schedules,
        )
        config_complex_after = Configuration(
            log_ids=log_ids,
            process_name=process,
            discovery_method=DiscoveryMethod.COMPLEX,
            timer_placement=TimerPlacement.AFTER,
            simulation_engine=SimulationEngine.PROSIMOS,
            working_schedules=working_schedules,
        )

        # --- Discover extraneous delays --- #
        # - Naive no hyperopt before
        naive_direct_before_enhancer = DirectEnhancer(event_log, simulation_model, config_naive_before)
        naive_direct_before_enhanced = naive_direct_before_enhancer.enhance_simulation_model_with_delays()
        _report_timers(evaluation_folder, "naive_direct_before_enhanced", naive_direct_before_enhancer)
        # - Complex no hyperopt before
        complex_direct_before_enhancer = DirectEnhancer(event_log, simulation_model, config_complex_before)
        complex_direct_before_enhanced = complex_direct_before_enhancer.enhance_simulation_model_with_delays()
        _report_timers(evaluation_folder, "complex_direct_before_enhanced", complex_direct_before_enhancer)
        # - Naive no hyperopt after
        naive_direct_after_enhancer = DirectEnhancer(event_log, simulation_model, config_naive_after)
        naive_direct_after_enhanced = naive_direct_after_enhancer.enhance_simulation_model_with_delays()
        _report_timers(evaluation_folder, "naive_direct_after_enhanced", naive_direct_after_enhancer)
        # - Complex no hyperopt after
        complex_direct_after_enhancer = DirectEnhancer(event_log, simulation_model, config_complex_after)
        complex_direct_after_enhanced = complex_direct_after_enhancer.enhance_simulation_model_with_delays()
        _report_timers(evaluation_folder, "complex_direct_after_enhanced", complex_direct_after_enhancer)

        # --- Write simulation models to file --- #
        _export_simulation_model(
            evaluation_folder, "{}_naive_direct_before_enhanced".format(process), naive_direct_before_enhanced
        )
        _export_simulation_model(
            evaluation_folder, "{}_complex_direct_before_enhanced".format(process), complex_direct_before_enhanced
        )
        _export_simulation_model(
            evaluation_folder, "{}_naive_direct_after_enhanced".format(process), naive_direct_after_enhanced
        )
        _export_simulation_model(
            evaluation_folder, "{}_complex_direct_after_enhanced".format(process), complex_direct_after_enhanced
        )


def _export_simulation_model(folder: Path, name: str, simulation_model: SimulationModel):
    simulation_model.bpmn_document.write(folder.joinpath(name + ".bpmn"), pretty_print=True)
    with open(folder.joinpath(name + ".json"), "w") as f:
        json.dump(simulation_model.simulation_parameters, f)


def _report_timers(folder: Path, name: str, enhancer: DirectEnhancer):
    with open(folder.joinpath(name + "_timers.txt"), "w") as output_file:
        for activity in enhancer.timers:
            output_file.write("'{}': {}\n".format(activity, enhancer.timers[activity]))


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
