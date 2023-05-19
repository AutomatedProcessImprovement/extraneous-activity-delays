import json
from pathlib import Path
from statistics import mean
from typing import Tuple

import pandas as pd
from lxml import etree

from extraneous_activity_delays.config import Configuration, TimerPlacement, SimulationModel, DiscoveryMethod, SimulationEngine, \
    OptimizationMetric
from extraneous_activity_delays.enhance_with_delays import DirectEnhancer, HyperOptEnhancer
from extraneous_activity_delays.utils.file_manager import create_folder
from pix_framework.calendar.prosimos_calendar import RCalendar
from pix_framework.input import read_csv_log
from pix_framework.log_ids import EventLogIDs

log_ids = EventLogIDs(
    case="case_id",
    activity="activity",
    resource="resource",
    start_time="start_time",
    end_time="end_time"
)
processes = [
    ("Insurance_Claims", "Insurance_Claims"),
    ("Insurance_Claims", "Insurance_Claims_1_timer"),
    ("Insurance_Claims", "Insurance_Claims_3_timers"),
    ("Insurance_Claims", "Insurance_Claims_5_timers"),
    ("Loan_Application", "Loan_Application"),
    ("Loan_Application", "Loan_Application_1_timer"),
    ("Loan_Application", "Loan_Application_3_timers"),
    ("Loan_Application", "Loan_Application_5_timers"),
    ("Pharmacy", "Pharmacy"),
    ("Pharmacy", "Pharmacy_1_timer"),
    ("Pharmacy", "Pharmacy_3_timers"),
    ("Pharmacy", "Pharmacy_5_timers"),
    ("Procure_to_Pay", "Procure_to_Pay"),
    ("Procure_to_Pay", "Procure_to_Pay_1_timer"),
    ("Procure_to_Pay", "Procure_to_Pay_3_timers"),
    ("Procure_to_Pay", "Procure_to_Pay_5_timers")
]


def inf_sys_evaluation():
    metrics_file_path = "../outputs/synthetic-evaluation/complete/metrics.csv"
    with open(metrics_file_path, 'a') as file:
        file.write("dataset,"
                   "naive_direct_precision,naive_direct_recall,naive_direct_sMAPE,"
                   "naive_hyperopt_precision,naive_hyperopt_recall,naive_hyperopt_sMAPE,"
                   "naive_hyperopt_holdout_precision,naive_hyperopt_holdout_recall,naive_hyperopt_holdout_sMAPE,"
                   "complex_direct_precision,complex_direct_recall,complex_direct_sMAPE,"
                   "complex_hyperopt_precision,complex_hyperopt_recall,complex_hyperopt_sMAPE,"
                   "complex_hyperopt_holdout_precision,complex_hyperopt_holdout_recall,complex_hyperopt_holdout_sMAPE"
                   "\n")
    # Run
    for no_timers_process, process in processes:
        # --- Raw paths --- #
        real_input_path = Configuration().PATH_INPUTS.joinpath("synthetic")
        simulation_bpmn_path = str(real_input_path.joinpath(no_timers_process + ".bpmn"))
        simulation_params_path = str(real_input_path.joinpath(no_timers_process + ".json"))
        log_path = str(real_input_path.joinpath(process + ".csv.gz"))

        # --- Evaluation folder --- #
        eval_folder = Configuration().PATH_OUTPUTS.joinpath("synthetic-evaluation").joinpath("complete").joinpath(process)
        create_folder(eval_folder)

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
        max_alpha = 10.0
        num_iterations = 100
        num_evaluation_simulations = 5
        config_naive = Configuration(
            log_ids=log_ids, process_name=process,
            max_alpha=max_alpha, num_iterations=num_iterations,
            num_evaluation_simulations=num_evaluation_simulations,
            discovery_method=DiscoveryMethod.NAIVE,
            timer_placement=TimerPlacement.BEFORE,
            simulation_engine=SimulationEngine.PROSIMOS,
            optimization_metric=OptimizationMetric.RELATIVE_EMD,
            working_schedules=working_schedules
        )
        config_complex = Configuration(
            log_ids=log_ids, process_name=process,
            max_alpha=max_alpha, num_iterations=num_iterations,
            num_evaluation_simulations=num_evaluation_simulations,
            discovery_method=DiscoveryMethod.COMPLEX,
            timer_placement=TimerPlacement.BEFORE,
            simulation_engine=SimulationEngine.PROSIMOS,
            optimization_metric=OptimizationMetric.RELATIVE_EMD,
            working_schedules=working_schedules
        )
        config_naive_holdout = Configuration(
            log_ids=log_ids, process_name=process,
            max_alpha=max_alpha, num_iterations=num_iterations,
            num_evaluation_simulations=num_evaluation_simulations,
            training_partition_ratio=0.5,
            discovery_method=DiscoveryMethod.NAIVE,
            timer_placement=TimerPlacement.BEFORE,
            simulation_engine=SimulationEngine.PROSIMOS,
            optimization_metric=OptimizationMetric.RELATIVE_EMD,
            working_schedules=working_schedules
        )
        config_complex_holdout = Configuration(
            log_ids=log_ids, process_name=process,
            max_alpha=max_alpha, num_iterations=num_iterations,
            num_evaluation_simulations=num_evaluation_simulations,
            training_partition_ratio=0.5,
            discovery_method=DiscoveryMethod.COMPLEX,
            timer_placement=TimerPlacement.BEFORE,
            simulation_engine=SimulationEngine.PROSIMOS,
            optimization_metric=OptimizationMetric.RELATIVE_EMD,
            working_schedules=working_schedules
        )

        # --- Discover extraneous delays --- #
        # - Naive no hyperopt
        naive_direct_enhancer = DirectEnhancer(event_log, simulation_model, config_naive)
        naive_direct_enhanced = naive_direct_enhancer.enhance_simulation_model_with_delays()
        # - Naive with hyperopt
        naive_hyperopt_enhancer = HyperOptEnhancer(event_log, simulation_model, config_naive)
        naive_hyperopt_enhanced = naive_hyperopt_enhancer.enhance_simulation_model_with_delays()
        # - Naive with hyperopt and holdout
        naive_hyperopt_holdout_enhancer = HyperOptEnhancer(event_log, simulation_model, config_naive_holdout)
        naive_hyperopt_holdout_enhanced = naive_hyperopt_holdout_enhancer.enhance_simulation_model_with_delays()
        # - Complex no hyperopt
        complex_direct_enhancer = DirectEnhancer(event_log, simulation_model, config_complex)
        complex_direct_enhanced = complex_direct_enhancer.enhance_simulation_model_with_delays()
        # - Complex with hyperopt
        complex_hyperopt_enhancer = HyperOptEnhancer(event_log, simulation_model, config_complex)
        complex_hyperopt_enhanced = complex_hyperopt_enhancer.enhance_simulation_model_with_delays()
        # - Complex with hyperopt and holdout
        complex_hyperopt_holdout_enhancer = HyperOptEnhancer(event_log, simulation_model, config_complex_holdout)
        complex_hyperopt_holdout_enhanced = complex_hyperopt_holdout_enhancer.enhance_simulation_model_with_delays()

        # --- Write simulation models to file --- #
        _export_simulation_model(eval_folder, "{}_naive_direct_enhanced".format(process), naive_direct_enhanced)
        _export_simulation_model(eval_folder, "{}_naive_hyperopt_enhanced".format(process), naive_hyperopt_enhanced)
        _export_simulation_model(eval_folder, "{}_naive_hyperopt_holdout_enhanced".format(process), naive_hyperopt_holdout_enhanced)
        _export_simulation_model(eval_folder, "{}_complex_direct_enhanced".format(process), complex_direct_enhanced)
        _export_simulation_model(eval_folder, "{}_complex_hyperopt_enhanced".format(process), complex_hyperopt_enhanced)
        _export_simulation_model(eval_folder, "{}_complex_hyperopt_holdout_enhanced".format(process), complex_hyperopt_holdout_enhanced)

        # --- Compute and report timer metrics --- #
        real_delays = {
            activity: list(events['extraneous_delay'])
            for activity, events in event_log.groupby(log_ids.activity)
            if (events['extraneous_delay'] > 0.0).any()
        }
        with open(metrics_file_path, 'a') as file:
            file.write("{},".format(process))
            precision, recall, smape = _compute_statistics(real_delays, naive_direct_enhancer.timers)
            file.write("{},{},{},".format(precision, recall, smape))
            precision, recall, smape = _compute_statistics(real_delays, naive_hyperopt_enhancer.timers)
            file.write("{},{},{},".format(precision, recall, smape))
            precision, recall, smape = _compute_statistics(real_delays, naive_hyperopt_holdout_enhancer.timers)
            file.write("{},{},{},".format(precision, recall, smape))
            precision, recall, smape = _compute_statistics(real_delays, complex_direct_enhancer.timers)
            file.write("{},{},{},".format(precision, recall, smape))
            precision, recall, smape = _compute_statistics(real_delays, complex_hyperopt_enhancer.timers)
            file.write("{},{},{},".format(precision, recall, smape))
            precision, recall, smape = _compute_statistics(real_delays, complex_hyperopt_holdout_enhancer.timers)
            file.write("{},{},{}\n".format(precision, recall, smape))


def _compute_statistics(real_delays: dict, estimated_timers: dict) -> Tuple[float, float, float]:
    if len(real_delays) > 0 and len(estimated_timers) > 0:
        # There are delays, and we discovered timers, compute
        precision = len([activity for activity in estimated_timers if activity in real_delays]) / len(estimated_timers)
        recall = len([activity for activity in estimated_timers if activity in real_delays]) / len(real_delays)
        smapes = []
        for activity in set(list(estimated_timers.keys()) + list(real_delays.keys())):
            if activity in real_delays and activity in estimated_timers:
                actual = real_delays[activity]
                forecast = estimated_timers[activity].generate_sample(len(actual))
                smapes += [2 * abs(mean(forecast) - mean(actual)) / (mean(actual) + mean(forecast))]
            else:
                smapes += [2.0]
        smape = mean(smapes)
    elif len(real_delays) == 0 and len(estimated_timers) == 0:
        # There are no delays, and we didn't discover timers, perfect
        precision, recall, smape = 1.0, 1.0, 0.0
    elif len(real_delays) == 0:
        # There are no delays, but we discovered timers
        precision, recall, smape = 0.0, float("nan"), 2.0
    else:
        # There are delays, but we didn't discover any timer
        precision, recall, smape = float("nan"), 0.0, 2.0

    return precision, recall, smape


def _compute_smape(event_log: pd.DataFrame) -> float:
    # Get activity instances with either estimated delay or actual delay
    estimated = event_log[(event_log['estimated_extraneous_delay'] > 0.0) | (event_log['extraneous_delay'] > 0.0)]
    # Compute smape
    if len(estimated) > 0:
        smape = sum([
            2 *
            abs(delays['estimated_extraneous_delay'] - delays['extraneous_delay']) /
            (delays['extraneous_delay'] + delays['estimated_extraneous_delay'])
            for index, delays
            in estimated[['estimated_extraneous_delay', 'extraneous_delay']].iterrows()
        ]) / len(estimated)
    else:
        smape = 0.0
    # Return value
    return smape


def _export_simulation_model(folder: Path, name: str, simulation_model: SimulationModel):
    simulation_model.bpmn_document.write(folder.joinpath(name + ".bpmn"), pretty_print=True)
    with open(folder.joinpath(name + ".json"), 'w') as f:
        json.dump(simulation_model.simulation_parameters, f)


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
            if int(resource['amount']) > 1:
                for i in range(int(resource['amount'])):
                    resource_calendars["{}_{}".format(resource['name'], i)] = calendars[resource['calendar']]
            else:
                resource_calendars[resource['name']] = calendars[resource['calendar']]
    # Return resource calendars
    return resource_calendars


if __name__ == '__main__':
    inf_sys_evaluation()
