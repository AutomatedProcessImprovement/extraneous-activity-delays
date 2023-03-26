import json
from pathlib import Path
from typing import Union, Tuple

import pandas as pd
from lxml import etree
from numpy import mean

from extraneous_activity_delays.config import Configuration, SimulationEngine, OptimizationMetric, SimulationModel, TimerPlacement, \
    DiscoveryMethod
from extraneous_activity_delays.enhance_with_delays import DirectEnhancer, HyperOptEnhancer
from extraneous_activity_delays.prosimos.simulator import simulate
from extraneous_activity_delays.utils.file_manager import create_folder
from log_similarity_metrics.absolute_event_distribution import absolute_event_distribution_distance
from log_similarity_metrics.relative_event_distribution import relative_event_distribution_distance
from pix_utils.calendar.resource_calendar import RCalendar
from pix_utils.input import read_csv_log
from pix_utils.log_ids import EventLogIDs

event_log_ids = EventLogIDs(
    case="case_id",
    activity="activity",
    resource="resource",
    start_time="start_time",
    end_time="end_time"
)


def inf_sys_evaluation():
    processes = ["AcademicCredentials"]
    metrics_file_path = "../outputs/real-life-evaluation/metrics.csv"
    with open(metrics_file_path, 'a') as file:
        file.write("dataset,"
                   "original_relative,naive_direct_relative,naive_hyperopt_relative,complex_direct_relative,complex_hyperopt_relative,"
                   "original_absolute,naive_direct_absolute,naive_hyperopt_absolute,complex_direct_absolute,complex_hyperopt_absolute"
                   "\n")
    # Run
    for process in processes:
        # --- Raw paths --- #
        real_input_path = Configuration().PATH_INPUTS.joinpath("real-life")
        simulation_model_bpmn_path = str(real_input_path.joinpath("prosimos-models").joinpath(process + ".bpmn"))
        simulation_model_params_path = str(real_input_path.joinpath("prosimos-models").joinpath(process + ".json"))
        train_log_path = str(real_input_path.joinpath(process + "_train.csv.gz"))
        test_log_path = str(real_input_path.joinpath(process + "_test.csv.gz"))

        # --- Evaluation folder --- #
        evaluation_folder = Configuration().PATH_OUTPUTS.joinpath("real-life-evaluation").joinpath(process)
        create_folder(evaluation_folder)

        # --- Read event logs --- #
        train_log = read_csv_log(train_log_path, event_log_ids)
        test_log = read_csv_log(test_log_path, event_log_ids)
        test_num_instances = len(test_log[event_log_ids.case].unique())
        test_start_time = min(test_log[event_log_ids.start_time])

        # --- Read simulation model --- #
        parser = etree.XMLParser(remove_blank_text=True)
        bpmn_model = etree.parse(simulation_model_bpmn_path, parser)
        with open(simulation_model_params_path) as json_file:
            simulation_parameters = json.load(json_file)
        simulation_model = SimulationModel(bpmn_model, simulation_parameters)
        working_schedules = _json_schedules_to_rcalendar(simulation_parameters)

        # --- Configurations --- #
        config_naive = Configuration(
            log_ids=event_log_ids, process_name=process,
            max_alpha=10.0, num_iterations=100,
            num_evaluation_simulations=10,
            discovery_method=DiscoveryMethod.NAIVE,
            timer_placement=TimerPlacement.AFTER,
            simulation_engine=SimulationEngine.PROSIMOS,
            optimization_metric=OptimizationMetric.RELATIVE_EMD,
            working_schedules=working_schedules
        )
        config_complex = Configuration(
            log_ids=event_log_ids, process_name=process,
            max_alpha=10.0, num_iterations=100,
            num_evaluation_simulations=10,
            discovery_method=DiscoveryMethod.COMPLEX,
            timer_placement=TimerPlacement.AFTER,
            simulation_engine=SimulationEngine.PROSIMOS,
            optimization_metric=OptimizationMetric.RELATIVE_EMD,
            working_schedules=working_schedules
        )

        # --- Discover extraneous delays --- #
        # - Naive no hyperopt
        naive_direct_enhancer = DirectEnhancer(train_log, simulation_model, config_naive)
        naive_direct_enhanced = naive_direct_enhancer.enhance_simulation_model_with_delays()
        _report_timers(evaluation_folder, "naive_direct_enhanced", naive_direct_enhancer)
        # - Naive with hyperopt
        naive_hyperopt_enhancer = HyperOptEnhancer(train_log, simulation_model, config_naive)
        naive_hyperopt_enhanced = naive_hyperopt_enhancer.enhance_simulation_model_with_delays()
        _report_timers(evaluation_folder, "naive_hyperopt_enhancer", naive_hyperopt_enhancer)
        # - Complex no hyperopt
        complex_direct_enhancer = DirectEnhancer(train_log, simulation_model, config_complex)
        complex_direct_enhanced = complex_direct_enhancer.enhance_simulation_model_with_delays()
        _report_timers(evaluation_folder, "complex_direct_enhanced", complex_direct_enhancer)
        # - Complex with hyperopt
        complex_hyperopt_enhancer = HyperOptEnhancer(train_log, simulation_model, config_complex)
        complex_hyperopt_enhanced = complex_hyperopt_enhancer.enhance_simulation_model_with_delays()
        _report_timers(evaluation_folder, "complex_hyperopt_enhancer", complex_hyperopt_enhancer)

        # --- Write simulation models to file --- #
        _export_simulation_model(evaluation_folder, "{}_original".format(process), simulation_model)
        _export_simulation_model(evaluation_folder, "{}_naive_direct_enhanced".format(process), naive_direct_enhanced)
        _export_simulation_model(evaluation_folder, "{}_naive_hyperopt_enhanced".format(process), naive_hyperopt_enhanced)
        _export_simulation_model(evaluation_folder, "{}_complex_direct_enhanced".format(process), complex_direct_enhanced)
        _export_simulation_model(evaluation_folder, "{}_complex_hyperopt_enhanced".format(process), complex_hyperopt_enhanced)

        # --- Simulate and Evaluate --- #
        # Set lists to store the results of each comparison and get the mean
        original_relative, original_absolute = [], []
        naive_direct_relative, naive_direct_absolute = [], []
        naive_hyperopt_relative, naive_hyperopt_absolute = [], []
        complex_direct_relative, complex_direct_absolute = [], []
        complex_hyperopt_relative, complex_hyperopt_absolute = [], []
        # Simulate many times and compute the mean
        for i in range(10):
            # Original
            relative, absolute = _simulate_and_evaluate(
                evaluation_folder, process, "original", i, test_num_instances, test_start_time, test_log
            )
            original_relative += [relative]
            original_absolute += [absolute]
            # Naive no hyperopt
            relative, absolute = _simulate_and_evaluate(
                evaluation_folder, process, "naive_direct_enhanced", i, test_num_instances, test_start_time, test_log
            )
            naive_direct_relative += [relative]
            naive_direct_absolute += [absolute]
            # Naive with hyperopt
            relative, absolute = _simulate_and_evaluate(
                evaluation_folder, process, "naive_hyperopt_enhanced", i, test_num_instances, test_start_time, test_log
            )
            naive_hyperopt_relative += [relative]
            naive_hyperopt_absolute += [absolute]
            # Complex no hyperopt
            relative, absolute = _simulate_and_evaluate(
                evaluation_folder, process, "complex_direct_enhanced", i, test_num_instances, test_start_time, test_log
            )
            complex_direct_relative += [relative]
            complex_direct_absolute += [absolute]
            # Complex with hyperopt
            relative, absolute = _simulate_and_evaluate(
                evaluation_folder, process, "complex_hyperopt_enhanced", i, test_num_instances, test_start_time, test_log
            )
            complex_hyperopt_relative += [relative]
            complex_hyperopt_absolute += [absolute]

        # --- Print results --- #
        with open(metrics_file_path, 'a') as output_file:
            output_file.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                process,
                mean(original_relative),
                mean(naive_direct_relative),
                mean(naive_hyperopt_relative),
                mean(complex_direct_relative),
                mean(complex_hyperopt_relative),
                mean(original_absolute),
                mean(naive_direct_absolute),
                mean(naive_hyperopt_absolute),
                mean(complex_direct_absolute),
                mean(complex_hyperopt_absolute)
            ))


def _simulate_and_evaluate(
        folder: Path, process: str, method: str, i: int, num_cases: int, start_timestamp: pd.Timestamp, test_log: pd.DataFrame
) -> Tuple[float, float]:
    # Simulate
    simulated_log_path = str(folder.joinpath("{}_sim_{}_{}.csv".format(process, method, i)))
    simulate(
        model_path=str(folder.joinpath("{}_{}.bpmn".format(process, method))),
        parameters_path=str(folder.joinpath("{}_{}.json".format(process, method))),
        num_cases=num_cases,
        starting_timestamp=start_timestamp,
        output_path=simulated_log_path
    )
    # Read simulated log
    original_simulated_event_log = read_csv_log(simulated_log_path, event_log_ids)
    # Evaluate simulated log
    relative = relative_event_distribution_distance(test_log, event_log_ids, original_simulated_event_log, event_log_ids)
    absolute = absolute_event_distribution_distance(test_log, event_log_ids, original_simulated_event_log, event_log_ids)
    # Return measures
    return relative, absolute


def _export_simulation_model(folder: Path, name: str, simulation_model: SimulationModel):
    simulation_model.bpmn_document.write(folder.joinpath(name + ".bpmn"), pretty_print=True)
    with open(folder.joinpath(name + ".json"), 'w') as f:
        json.dump(simulation_model.simulation_parameters, f)


def _report_timers(folder: Path, name: str, enhancer: Union[DirectEnhancer, HyperOptEnhancer]):
    with open(folder.joinpath(name + "_timers.txt"), 'w') as output_file:
        if type(enhancer) is DirectEnhancer:
            # Print timers
            for activity in enhancer.best_timers:
                output_file.write("'{}': {}\n".format(activity, enhancer.timers[activity]))
        elif type(enhancer) is HyperOptEnhancer:
            # Print best timers and losses
            for activity in enhancer.best_timers:
                output_file.write("'{}': {}\n".format(activity, enhancer.best_timers[activity]))
            output_file.write("\nLosses: {}".format(enhancer.losses))


def _json_schedules_to_rcalendar(simulation_parameters: dict) -> dict:
    """
    Transform the calendars specified as part of the simulation parameters to a dict with the ID of the resources as key, and their
    calendar (RCalendar) as value.

    :param simulation_parameters: dictionary with the parameters for prosimos simulation.

    :return: a dict with the ID of the resources as key and their calendar as value.
    """
    # Read calendars
    calendars = {}
    for calendar in simulation_parameters['resource_calendars']:
        r_calendar = RCalendar(calendar["id"])
        for slot in calendar["time_periods"]:
            r_calendar.add_calendar_item(
                slot["from"], slot["to"], slot["beginTime"], slot["endTime"]
            )
        calendars[r_calendar.calendar_id] = r_calendar
    # Assign calendars to each resource
    resource_calendars = {}
    for profile in simulation_parameters['resource_profiles']:
        for resource in profile['resource_list']:
            resource_calendars[resource['id']] = calendars[resource['calendar']]
    # Return resource calendars
    return resource_calendars


if __name__ == '__main__':
    inf_sys_evaluation()
