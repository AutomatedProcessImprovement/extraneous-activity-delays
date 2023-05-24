import json
import time
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import pandas as pd
from lxml import etree
from scipy.stats import t

from extraneous_activity_delays.config import (
    Configuration,
    SimulationEngine,
    OptimizationMetric,
    SimulationModel,
    TimerPlacement,
    DiscoveryMethod,
)
from extraneous_activity_delays.enhance_with_delays import DirectEnhancer, HyperOptEnhancer
from extraneous_activity_delays.prosimos.simulator import simulate
from extraneous_activity_delays.utils.file_manager import create_folder
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
from pix_framework.calendar.resource_calendar import RCalendar
from pix_framework.input import read_csv_log
from pix_framework.log_ids import EventLogIDs

event_log_ids = EventLogIDs(
    case="case_id", activity="activity", resource="resource", start_time="start_time", end_time="end_time"
)


def inf_sys_evaluation():
    processes = ["AcademicCredentials", "BPIC_2012_W", "BPIC_2017_W"]
    metrics_file_path = "../outputs/real-life-evaluation/metrics.csv"
    metrics_ct_file_path = "../outputs/real-life-evaluation/metrics-cycle-time.csv"
    with open(metrics_file_path, "a") as file:
        file.write("name,relative_mean,relative_cnf,absolute_mean,absolute_cnf,runtime\n")
    with open(metrics_ct_file_path, "a") as file:
        file.write(
            "name,min_mean,min_cnf,q1_mean,q1_cnf,median_mean,median_cnf,mean_mean,mean_cnf,q3_mean,q3_cnf,max_mean,max_cnf\n"
        )
    # Run
    for process in processes:
        # --- Raw paths --- #
        real_input_path = Configuration().PATH_INPUTS.joinpath("real-life")
        simulation_model_bpmn_path = str(real_input_path.joinpath("prosimos-models").joinpath(process + ".bpmn"))
        simulation_model_params_path = str(real_input_path.joinpath("prosimos-models").joinpath(process + ".json"))
        train_log_path = str(real_input_path.joinpath(process + "_train.csv.gz"))
        test_log_path = str(real_input_path.joinpath(process + "_test.csv.gz"))

        # --- Evaluation folder --- #
        eval_folder = Configuration().PATH_OUTPUTS.joinpath("real-life-evaluation").joinpath(process)
        create_folder(eval_folder)

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
        max_alpha = 10.0
        num_iterations = 100
        num_evaluation_simulations = 5
        config_naive_before = Configuration(
            log_ids=event_log_ids,
            process_name=process,
            max_alpha=max_alpha,
            num_iterations=num_iterations,
            num_evaluation_simulations=num_evaluation_simulations,
            discovery_method=DiscoveryMethod.NAIVE,
            timer_placement=TimerPlacement.BEFORE,
            simulation_engine=SimulationEngine.PROSIMOS,
            optimization_metric=OptimizationMetric.RELATIVE_EMD,
            working_schedules=working_schedules,
        )
        config_complex_before = Configuration(
            log_ids=event_log_ids,
            process_name=process,
            max_alpha=max_alpha,
            num_iterations=num_iterations,
            num_evaluation_simulations=num_evaluation_simulations,
            discovery_method=DiscoveryMethod.COMPLEX,
            timer_placement=TimerPlacement.BEFORE,
            simulation_engine=SimulationEngine.PROSIMOS,
            optimization_metric=OptimizationMetric.RELATIVE_EMD,
            working_schedules=working_schedules,
        )
        config_naive_after = Configuration(
            log_ids=event_log_ids,
            process_name=process,
            max_alpha=max_alpha,
            num_iterations=num_iterations,
            num_evaluation_simulations=num_evaluation_simulations,
            discovery_method=DiscoveryMethod.NAIVE,
            timer_placement=TimerPlacement.AFTER,
            simulation_engine=SimulationEngine.PROSIMOS,
            optimization_metric=OptimizationMetric.RELATIVE_EMD,
            working_schedules=working_schedules,
        )
        config_complex_after = Configuration(
            log_ids=event_log_ids,
            process_name=process,
            max_alpha=max_alpha,
            num_iterations=num_iterations,
            num_evaluation_simulations=num_evaluation_simulations,
            discovery_method=DiscoveryMethod.COMPLEX,
            timer_placement=TimerPlacement.AFTER,
            simulation_engine=SimulationEngine.PROSIMOS,
            optimization_metric=OptimizationMetric.RELATIVE_EMD,
            working_schedules=working_schedules,
        )
        config_naive_holdout_before = Configuration(
            log_ids=event_log_ids,
            process_name=process,
            max_alpha=max_alpha,
            num_iterations=num_iterations,
            num_evaluation_simulations=num_evaluation_simulations,
            training_partition_ratio=0.5,
            discovery_method=DiscoveryMethod.NAIVE,
            timer_placement=TimerPlacement.BEFORE,
            simulation_engine=SimulationEngine.PROSIMOS,
            optimization_metric=OptimizationMetric.RELATIVE_EMD,
            working_schedules=working_schedules,
        )
        config_complex_holdout_before = Configuration(
            log_ids=event_log_ids,
            process_name=process,
            max_alpha=max_alpha,
            num_iterations=num_iterations,
            num_evaluation_simulations=num_evaluation_simulations,
            training_partition_ratio=0.5,
            discovery_method=DiscoveryMethod.COMPLEX,
            timer_placement=TimerPlacement.BEFORE,
            simulation_engine=SimulationEngine.PROSIMOS,
            optimization_metric=OptimizationMetric.RELATIVE_EMD,
            working_schedules=working_schedules,
        )
        config_naive_holdout_after = Configuration(
            log_ids=event_log_ids,
            process_name=process,
            max_alpha=max_alpha,
            num_iterations=num_iterations,
            num_evaluation_simulations=num_evaluation_simulations,
            training_partition_ratio=0.5,
            discovery_method=DiscoveryMethod.NAIVE,
            timer_placement=TimerPlacement.AFTER,
            simulation_engine=SimulationEngine.PROSIMOS,
            optimization_metric=OptimizationMetric.RELATIVE_EMD,
            working_schedules=working_schedules,
        )
        config_complex_holdout_after = Configuration(
            log_ids=event_log_ids,
            process_name=process,
            max_alpha=max_alpha,
            num_iterations=num_iterations,
            num_evaluation_simulations=num_evaluation_simulations,
            training_partition_ratio=0.5,
            discovery_method=DiscoveryMethod.COMPLEX,
            timer_placement=TimerPlacement.AFTER,
            simulation_engine=SimulationEngine.PROSIMOS,
            optimization_metric=OptimizationMetric.RELATIVE_EMD,
            working_schedules=working_schedules,
        )

        # --- Discover extraneous delays --- #
        # -- Timer Placement: BEFORE -- #
        # - Naive no hyperopt
        runtime_start = time.time()
        naive_direct_before_enhancer = DirectEnhancer(train_log, simulation_model, config_naive_before)
        naive_direct_before_enhanced = naive_direct_before_enhancer.enhance_simulation_model_with_delays()
        runtime_naive_direct_before = time.time() - runtime_start
        _report_timers(eval_folder, "naive_direct_before_enhanced", naive_direct_before_enhancer)
        # - Naive with hyperopt
        runtime_start = time.time()
        naive_hyperopt_before_enhancer = HyperOptEnhancer(train_log, simulation_model, config_naive_before)
        naive_hyperopt_before_enhanced = naive_hyperopt_before_enhancer.enhance_simulation_model_with_delays()
        runtime_naive_hyperopt_before = time.time() - runtime_start
        _report_timers(eval_folder, "naive_hyperopt_before_enhancer", naive_hyperopt_before_enhancer)
        # - Naive with hyperopt and holdout
        runtime_start = time.time()
        naive_hyperopt_holdout_before_enhancer = HyperOptEnhancer(
            train_log, simulation_model, config_naive_holdout_before
        )
        naive_hyperopt_holdout_before_enhanced = (
            naive_hyperopt_holdout_before_enhancer.enhance_simulation_model_with_delays()
        )
        runtime_naive_hyperopt_holdout_before = time.time() - runtime_start
        _report_timers(eval_folder, "naive_hyperopt_holdout_before_enhancer", naive_hyperopt_holdout_before_enhancer)
        # - Complex no hyperopt
        runtime_start = time.time()
        complex_direct_before_enhancer = DirectEnhancer(train_log, simulation_model, config_complex_before)
        complex_direct_before_enhanced = complex_direct_before_enhancer.enhance_simulation_model_with_delays()
        runtime_complex_direct_before = time.time() - runtime_start
        _report_timers(eval_folder, "complex_direct_before_enhanced", complex_direct_before_enhancer)
        # - Complex with hyperopt
        runtime_start = time.time()
        complex_hyperopt_before_enhancer = HyperOptEnhancer(train_log, simulation_model, config_complex_before)
        complex_hyperopt_before_enhanced = complex_hyperopt_before_enhancer.enhance_simulation_model_with_delays()
        runtime_complex_hyperopt_before = time.time() - runtime_start
        _report_timers(eval_folder, "complex_hyperopt_before_enhancer", complex_hyperopt_before_enhancer)
        # - Complex with hyperopt and holdout
        runtime_start = time.time()
        complex_hyperopt_holdout_before_enhancer = HyperOptEnhancer(
            train_log, simulation_model, config_complex_holdout_before
        )
        complex_hyperopt_holdout_before_enhanced = (
            complex_hyperopt_holdout_before_enhancer.enhance_simulation_model_with_delays()
        )
        runtime_complex_hyperopt_holdout_before = time.time() - runtime_start
        _report_timers(
            eval_folder, "complex_hyperopt_holdout_before_enhancer", complex_hyperopt_holdout_before_enhancer
        )

        # -- Timer Placement: AFTER -- #
        # - Naive no hyperopt
        runtime_start = time.time()
        naive_direct_after_enhancer = DirectEnhancer(train_log, simulation_model, config_naive_after)
        naive_direct_after_enhanced = naive_direct_after_enhancer.enhance_simulation_model_with_delays()
        runtime_naive_direct_after = time.time() - runtime_start
        _report_timers(eval_folder, "naive_direct_after_enhanced", naive_direct_after_enhancer)
        # - Naive with hyperopt
        runtime_start = time.time()
        naive_hyperopt_after_enhancer = HyperOptEnhancer(train_log, simulation_model, config_naive_after)
        naive_hyperopt_after_enhanced = naive_hyperopt_after_enhancer.enhance_simulation_model_with_delays()
        runtime_naive_hyperopt_after = time.time() - runtime_start
        _report_timers(eval_folder, "naive_hyperopt_after_enhancer", naive_hyperopt_after_enhancer)
        # - Naive with hyperopt and holdout
        runtime_start = time.time()
        naive_hyperopt_holdout_after_enhancer = HyperOptEnhancer(
            train_log, simulation_model, config_naive_holdout_after
        )
        naive_hyperopt_holdout_after_enhanced = (
            naive_hyperopt_holdout_after_enhancer.enhance_simulation_model_with_delays()
        )
        runtime_naive_hyperopt_holdout_after = time.time() - runtime_start
        _report_timers(eval_folder, "naive_hyperopt_holdout_after_enhancer", naive_hyperopt_holdout_after_enhancer)
        # - Complex no hyperopt
        runtime_start = time.time()
        complex_direct_after_enhancer = DirectEnhancer(train_log, simulation_model, config_complex_after)
        complex_direct_after_enhanced = complex_direct_after_enhancer.enhance_simulation_model_with_delays()
        runtime_complex_direct_after = time.time() - runtime_start
        _report_timers(eval_folder, "complex_direct_after_enhanced", complex_direct_after_enhancer)
        # - Complex with hyperopt
        runtime_start = time.time()
        complex_hyperopt_after_enhancer = HyperOptEnhancer(train_log, simulation_model, config_complex_after)
        complex_hyperopt_after_enhanced = complex_hyperopt_after_enhancer.enhance_simulation_model_with_delays()
        runtime_complex_hyperopt_after = time.time() - runtime_start
        _report_timers(eval_folder, "complex_hyperopt_after_enhancer", complex_hyperopt_after_enhancer)
        # - Complex with hyperopt and holdout
        runtime_start = time.time()
        complex_hyperopt_holdout_after_enhancer = HyperOptEnhancer(
            train_log, simulation_model, config_complex_holdout_after
        )
        complex_hyperopt_holdout_after_enhanced = (
            complex_hyperopt_holdout_after_enhancer.enhance_simulation_model_with_delays()
        )
        runtime_complex_hyperopt_holdout_after = time.time() - runtime_start
        _report_timers(eval_folder, "complex_hyperopt_holdout_after_enhancer", complex_hyperopt_holdout_after_enhancer)

        # --- Write simulation models to file --- #
        export_sm(eval_folder, "{}_original".format(process), simulation_model)
        export_sm(eval_folder, "{}_naive_direct_before_enhanced".format(process), naive_direct_before_enhanced)
        export_sm(eval_folder, "{}_naive_hyperopt_before_enhanced".format(process), naive_hyperopt_before_enhanced)
        export_sm(
            eval_folder,
            "{}_naive_hyperopt_holdout_before_enhanced".format(process),
            naive_hyperopt_holdout_before_enhanced,
        )
        export_sm(eval_folder, "{}_complex_direct_before_enhanced".format(process), complex_direct_before_enhanced)
        export_sm(eval_folder, "{}_complex_hyperopt_before_enhanced".format(process), complex_hyperopt_before_enhanced)
        export_sm(
            eval_folder,
            "{}_complex_hyperopt_holdout_before_enhanced".format(process),
            complex_hyperopt_holdout_before_enhanced,
        )
        export_sm(eval_folder, "{}_naive_direct_after_enhanced".format(process), naive_direct_after_enhanced)
        export_sm(eval_folder, "{}_naive_hyperopt_after_enhanced".format(process), naive_hyperopt_after_enhanced)
        export_sm(
            eval_folder,
            "{}_naive_hyperopt_holdout_after_enhanced".format(process),
            naive_hyperopt_holdout_after_enhanced,
        )
        export_sm(eval_folder, "{}_complex_direct_after_enhanced".format(process), complex_direct_after_enhanced)
        export_sm(eval_folder, "{}_complex_hyperopt_after_enhanced".format(process), complex_hyperopt_after_enhanced)
        export_sm(
            eval_folder,
            "{}_complex_hyperopt_holdout_after_enhanced".format(process),
            complex_hyperopt_holdout_after_enhanced,
        )

        # --- Simulate and Evaluate --- #
        # Set lists to store the results of each comparison and get the mean
        original_relative, original_absolute, original_cts = [], [], []
        naive_direct_before_relative, naive_direct_before_absolute, naive_direct_before_cts = [], [], []
        naive_hyperopt_before_relative, naive_hyperopt_before_absolute, naive_hyperopt_before_cts = [], [], []
        (
            naive_hyperopt_holdout_before_relative,
            naive_hyperopt_holdout_before_absolute,
            naive_hyperopt_holdout_before_cts,
        ) = ([], [], [])
        complex_direct_before_relative, complex_direct_before_absolute, complex_direct_before_cts = [], [], []
        complex_hyperopt_before_relative, complex_hyperopt_before_absolute, complex_hyperopt_before_cts = [], [], []
        (
            complex_hyperopt_holdout_before_relative,
            complex_hyperopt_holdout_before_absolute,
            complex_hyperopt_holdout_before_cts,
        ) = ([], [], [])
        naive_direct_after_relative, naive_direct_after_absolute, naive_direct_after_cts = [], [], []
        naive_hyperopt_after_relative, naive_hyperopt_after_absolute, naive_hyperopt_after_cts = [], [], []
        (
            naive_hyperopt_holdout_after_relative,
            naive_hyperopt_holdout_after_absolute,
            naive_hyperopt_holdout_after_cts,
        ) = ([], [], [])
        complex_direct_after_relative, complex_direct_after_absolute, complex_direct_after_cts = [], [], []
        complex_hyperopt_after_relative, complex_hyperopt_after_absolute, complex_hyperopt_after_cts = [], [], []
        (
            complex_hyperopt_holdout_after_relative,
            complex_hyperopt_holdout_after_absolute,
            complex_hyperopt_holdout_after_cts,
        ) = ([], [], [])
        # Simulate many times and compute the mean
        for i in range(10):
            # Original
            relative, absolute, cycle_times = _simulate_and_evaluate(
                eval_folder, process, "original", i, test_num_instances, test_start_time, test_log
            )
            original_relative += [relative]
            original_absolute += [absolute]
            original_cts += [cycle_times]

            # -- Timer Placement: BEFORE -- #
            # Naive no hyperopt
            relative, absolute, cycle_times = _simulate_and_evaluate(
                eval_folder, process, "naive_direct_before_enhanced", i, test_num_instances, test_start_time, test_log
            )
            naive_direct_before_relative += [relative]
            naive_direct_before_absolute += [absolute]
            naive_direct_before_cts += [cycle_times]
            # Naive with hyperopt
            relative, absolute, cycle_times = _simulate_and_evaluate(
                eval_folder, process, "naive_hyperopt_before_enhanced", i, test_num_instances, test_start_time, test_log
            )
            naive_hyperopt_before_relative += [relative]
            naive_hyperopt_before_absolute += [absolute]
            naive_hyperopt_before_cts += [cycle_times]
            # Naive with hyperopt and holdout
            relative, absolute, cycle_times = _simulate_and_evaluate(
                eval_folder,
                process,
                "naive_hyperopt_holdout_before_enhanced",
                i,
                test_num_instances,
                test_start_time,
                test_log,
            )
            naive_hyperopt_holdout_before_relative += [relative]
            naive_hyperopt_holdout_before_absolute += [absolute]
            naive_hyperopt_holdout_before_cts += [cycle_times]
            # Complex no hyperopt
            relative, absolute, cycle_times = _simulate_and_evaluate(
                eval_folder, process, "complex_direct_before_enhanced", i, test_num_instances, test_start_time, test_log
            )
            complex_direct_before_relative += [relative]
            complex_direct_before_absolute += [absolute]
            complex_direct_before_cts += [cycle_times]
            # Complex with hyperopt
            relative, absolute, cycle_times = _simulate_and_evaluate(
                eval_folder,
                process,
                "complex_hyperopt_before_enhanced",
                i,
                test_num_instances,
                test_start_time,
                test_log,
            )
            complex_hyperopt_before_relative += [relative]
            complex_hyperopt_before_absolute += [absolute]
            complex_hyperopt_before_cts += [cycle_times]
            # Complex with hyperopt and holdout
            relative, absolute, cycle_times = _simulate_and_evaluate(
                eval_folder,
                process,
                "complex_hyperopt_holdout_before_enhanced",
                i,
                test_num_instances,
                test_start_time,
                test_log,
            )
            complex_hyperopt_holdout_before_relative += [relative]
            complex_hyperopt_holdout_before_absolute += [absolute]
            complex_hyperopt_holdout_before_cts += [cycle_times]

            # -- Timer Placement: AFTER -- #
            # Naive no hyperopt
            relative, absolute, cycle_times = _simulate_and_evaluate(
                eval_folder, process, "naive_direct_after_enhanced", i, test_num_instances, test_start_time, test_log
            )
            naive_direct_after_relative += [relative]
            naive_direct_after_absolute += [absolute]
            naive_direct_after_cts += [cycle_times]
            # Naive with hyperopt
            relative, absolute, cycle_times = _simulate_and_evaluate(
                eval_folder, process, "naive_hyperopt_after_enhanced", i, test_num_instances, test_start_time, test_log
            )
            naive_hyperopt_after_relative += [relative]
            naive_hyperopt_after_absolute += [absolute]
            naive_hyperopt_after_cts += [cycle_times]
            # Naive with hyperopt and holdout
            relative, absolute, cycle_times = _simulate_and_evaluate(
                eval_folder,
                process,
                "naive_hyperopt_holdout_after_enhanced",
                i,
                test_num_instances,
                test_start_time,
                test_log,
            )
            naive_hyperopt_holdout_after_relative += [relative]
            naive_hyperopt_holdout_after_absolute += [absolute]
            naive_hyperopt_holdout_after_cts += [cycle_times]
            # Complex no hyperopt
            relative, absolute, cycle_times = _simulate_and_evaluate(
                eval_folder, process, "complex_direct_after_enhanced", i, test_num_instances, test_start_time, test_log
            )
            complex_direct_after_relative += [relative]
            complex_direct_after_absolute += [absolute]
            complex_direct_after_cts += [cycle_times]
            # Complex with hyperopt
            relative, absolute, cycle_times = _simulate_and_evaluate(
                eval_folder,
                process,
                "complex_hyperopt_after_enhanced",
                i,
                test_num_instances,
                test_start_time,
                test_log,
            )
            complex_hyperopt_after_relative += [relative]
            complex_hyperopt_after_absolute += [absolute]
            complex_hyperopt_after_cts += [cycle_times]
            # Complex with hyperopt
            relative, absolute, cycle_times = _simulate_and_evaluate(
                eval_folder,
                process,
                "complex_hyperopt_holdout_after_enhanced",
                i,
                test_num_instances,
                test_start_time,
                test_log,
            )
            complex_hyperopt_holdout_after_relative += [relative]
            complex_hyperopt_holdout_after_absolute += [absolute]
            complex_hyperopt_holdout_after_cts += [cycle_times]

        # --- Print results --- #
        with open(metrics_file_path, "a") as output_file:
            # Original
            relative_avg, relative_cnf = compute_mean_conf_interval(original_relative)
            absolute_avg, absolute_cnf = compute_mean_conf_interval(original_absolute)
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    "{}_original".format(process), relative_avg, relative_cnf, absolute_avg, absolute_cnf, 0.0
                )
            )
            # Naive Direct Before
            relative_avg, relative_cnf = compute_mean_conf_interval(naive_direct_before_relative)
            absolute_avg, absolute_cnf = compute_mean_conf_interval(naive_direct_before_absolute)
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    "{}_naive_direct_before".format(process),
                    relative_avg,
                    relative_cnf,
                    absolute_avg,
                    absolute_cnf,
                    runtime_naive_direct_before,
                )
            )
            # Naive Hyperopt Before
            relative_avg, relative_cnf = compute_mean_conf_interval(naive_hyperopt_before_relative)
            absolute_avg, absolute_cnf = compute_mean_conf_interval(naive_hyperopt_before_absolute)
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    "{}_naive_hyperopt_before".format(process),
                    relative_avg,
                    relative_cnf,
                    absolute_avg,
                    absolute_cnf,
                    runtime_naive_hyperopt_before,
                )
            )
            # Naive Hyperopt and Holdout Before
            relative_avg, relative_cnf = compute_mean_conf_interval(naive_hyperopt_holdout_before_relative)
            absolute_avg, absolute_cnf = compute_mean_conf_interval(naive_hyperopt_holdout_before_absolute)
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    "{}_naive_hyperopt_holdout_before".format(process),
                    relative_avg,
                    relative_cnf,
                    absolute_avg,
                    absolute_cnf,
                    runtime_naive_hyperopt_holdout_before,
                )
            )
            # Complex Direct Before
            relative_avg, relative_cnf = compute_mean_conf_interval(complex_direct_before_relative)
            absolute_avg, absolute_cnf = compute_mean_conf_interval(complex_direct_before_absolute)
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    "{}_complex_direct_before".format(process),
                    relative_avg,
                    relative_cnf,
                    absolute_avg,
                    absolute_cnf,
                    runtime_complex_direct_before,
                )
            )
            # Complex Hyperopt Before
            relative_avg, relative_cnf = compute_mean_conf_interval(complex_hyperopt_before_relative)
            absolute_avg, absolute_cnf = compute_mean_conf_interval(complex_hyperopt_before_absolute)
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    "{}_complex_hyperopt_before".format(process),
                    relative_avg,
                    relative_cnf,
                    absolute_avg,
                    absolute_cnf,
                    runtime_complex_hyperopt_before,
                )
            )
            # Complex Hyperopt and Holdout Before
            relative_avg, relative_cnf = compute_mean_conf_interval(complex_hyperopt_holdout_before_relative)
            absolute_avg, absolute_cnf = compute_mean_conf_interval(complex_hyperopt_holdout_before_absolute)
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    "{}_complex_hyperopt_holdout_before".format(process),
                    relative_avg,
                    relative_cnf,
                    absolute_avg,
                    absolute_cnf,
                    runtime_complex_hyperopt_holdout_before,
                )
            )
            # Naive Direct After
            relative_avg, relative_cnf = compute_mean_conf_interval(naive_direct_after_relative)
            absolute_avg, absolute_cnf = compute_mean_conf_interval(naive_direct_after_absolute)
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    "{}_naive_direct_after".format(process),
                    relative_avg,
                    relative_cnf,
                    absolute_avg,
                    absolute_cnf,
                    runtime_naive_direct_after,
                )
            )
            # Naive Hyperopt After
            relative_avg, relative_cnf = compute_mean_conf_interval(naive_hyperopt_after_relative)
            absolute_avg, absolute_cnf = compute_mean_conf_interval(naive_hyperopt_after_absolute)
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    "{}_naive_hyperopt_after".format(process),
                    relative_avg,
                    relative_cnf,
                    absolute_avg,
                    absolute_cnf,
                    runtime_naive_hyperopt_after,
                )
            )
            # Naive Hyperopt and Holdout After
            relative_avg, relative_cnf = compute_mean_conf_interval(naive_hyperopt_holdout_after_relative)
            absolute_avg, absolute_cnf = compute_mean_conf_interval(naive_hyperopt_holdout_after_absolute)
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    "{}_naive_hyperopt_holdout_after".format(process),
                    relative_avg,
                    relative_cnf,
                    absolute_avg,
                    absolute_cnf,
                    runtime_naive_hyperopt_holdout_after,
                )
            )
            # Complex Direct After
            relative_avg, relative_cnf = compute_mean_conf_interval(complex_direct_after_relative)
            absolute_avg, absolute_cnf = compute_mean_conf_interval(complex_direct_after_absolute)
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    "{}_complex_direct_after".format(process),
                    relative_avg,
                    relative_cnf,
                    absolute_avg,
                    absolute_cnf,
                    runtime_complex_direct_after,
                )
            )
            # Complex Hyperopt After
            relative_avg, relative_cnf = compute_mean_conf_interval(complex_hyperopt_after_relative)
            absolute_avg, absolute_cnf = compute_mean_conf_interval(complex_hyperopt_after_absolute)
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    "{}_complex_hyperopt_after".format(process),
                    relative_avg,
                    relative_cnf,
                    absolute_avg,
                    absolute_cnf,
                    runtime_complex_hyperopt_after,
                )
            )
            # Complex Hyperopt and Holdout After
            relative_avg, relative_cnf = compute_mean_conf_interval(complex_hyperopt_holdout_after_relative)
            absolute_avg, absolute_cnf = compute_mean_conf_interval(complex_hyperopt_holdout_after_absolute)
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    "{}_complex_hyperopt_holdout_after".format(process),
                    relative_avg,
                    relative_cnf,
                    absolute_avg,
                    absolute_cnf,
                    runtime_complex_hyperopt_holdout_after,
                )
            )
        with open(metrics_ct_file_path, "a") as output_file:
            # Train log
            cycle_times = compute_cycle_time_stats(train_log, event_log_ids)
            output_file.write(
                "{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    "{}_train_log".format(process),
                    cycle_times["min"],
                    0.0,
                    cycle_times["q1"],
                    0.0,
                    cycle_times["median"],
                    0.0,
                    cycle_times["mean"],
                    0.0,
                    cycle_times["q3"],
                    0.0,
                    cycle_times["max"],
                    0.0,
                )
            )
            # Test log
            cycle_times = compute_cycle_time_stats(test_log, event_log_ids)
            output_file.write(
                "{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    "{}_test_log".format(process),
                    cycle_times["min"],
                    0.0,
                    cycle_times["q1"],
                    0.0,
                    cycle_times["median"],
                    0.0,
                    cycle_times["mean"],
                    0.0,
                    cycle_times["q3"],
                    0.0,
                    cycle_times["max"],
                    0.0,
                )
            )
            # Original
            formatted_output = format_output_cycle_times(process, "original", original_cts)
            output_file.write(formatted_output)
            # Naive Direct Before
            formatted_output = format_output_cycle_times(process, "naive_direct_before", naive_direct_before_cts)
            output_file.write(formatted_output)
            # Naive Hyperopt Before
            formatted_output = format_output_cycle_times(process, "naive_hyperopt_before", naive_hyperopt_before_cts)
            output_file.write(formatted_output)
            # Naive Hyperopt and Holdout Before
            formatted_output = format_output_cycle_times(
                process, "naive_hyperopt_holdout_before", naive_hyperopt_holdout_before_cts
            )
            output_file.write(formatted_output)
            # Complex Direct Before
            formatted_output = format_output_cycle_times(process, "complex_direct_before", complex_direct_before_cts)
            output_file.write(formatted_output)
            # Complex Hyperopt Before
            formatted_output = format_output_cycle_times(
                process, "complex_hyperopt_before", complex_hyperopt_before_cts
            )
            output_file.write(formatted_output)
            # Complex Hyperopt and Holdout Before
            formatted_output = format_output_cycle_times(
                process, "complex_hyperopt_holdout_before", complex_hyperopt_holdout_before_cts
            )
            output_file.write(formatted_output)
            # Naive Direct After
            formatted_output = format_output_cycle_times(process, "naive_direct_after", naive_direct_after_cts)
            output_file.write(formatted_output)
            # Naive Hyperopt After
            formatted_output = format_output_cycle_times(process, "naive_hyperopt_after", naive_hyperopt_after_cts)
            output_file.write(formatted_output)
            # Naive Hyperopt and Holdout After
            formatted_output = format_output_cycle_times(
                process, "naive_hyperopt_holdout_after", naive_hyperopt_holdout_after_cts
            )
            output_file.write(formatted_output)
            # Complex Direct After
            formatted_output = format_output_cycle_times(process, "complex_direct_after", complex_direct_after_cts)
            output_file.write(formatted_output)
            # Complex Hyperopt After
            formatted_output = format_output_cycle_times(process, "complex_hyperopt_after", complex_hyperopt_after_cts)
            output_file.write(formatted_output)
            # Complex Hyperopt and Holdout After
            formatted_output = format_output_cycle_times(
                process, "complex_hyperopt_holdout_after", complex_hyperopt_holdout_after_cts
            )
            output_file.write(formatted_output)


def format_output_cycle_times(process: str, method: str, cycle_times: list) -> str:
    min_average, min_cnf = compute_mean_conf_interval([ct["min"] for ct in cycle_times])
    q1_average, q1_cnf = compute_mean_conf_interval([ct["q1"] for ct in cycle_times])
    median_average, median_cnf = compute_mean_conf_interval([ct["median"] for ct in cycle_times])
    mean_average, mean_cnf = compute_mean_conf_interval([ct["mean"] for ct in cycle_times])
    q3_average, q3_cnf = compute_mean_conf_interval([ct["q3"] for ct in cycle_times])
    max_average, max_cnf = compute_mean_conf_interval([ct["max"] for ct in cycle_times])
    return "{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
        "{}_{}".format(process, method),
        min_average,
        min_cnf,
        q1_average,
        q1_cnf,
        median_average,
        median_cnf,
        mean_average,
        mean_cnf,
        q3_average,
        q3_cnf,
        max_average,
        max_cnf,
    )


def compute_mean_conf_interval(data: list, confidence: float = 0.95) -> Tuple[float, float]:
    # Compute the sample mean and standard deviation
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # ddof=1 calculates the sample standard deviation
    # Compute the degrees of freedom
    df = len(data) - 1
    # Compute the t-value for the confidence level
    t_value = t.ppf(1 - (1 - confidence) / 2, df)
    # Compute the standard error of the mean
    std_error = sample_std / np.sqrt(len(data))
    conf_interval = t_value * std_error
    # Compute the confidence interval
    return sample_mean, conf_interval


def compute_cycle_time_stats(event_log: pd.DataFrame, log_ids: EventLogIDs):
    cycle_times = []
    for case, events in event_log.groupby(log_ids.case):
        cycle_times += [(events[log_ids.end_time].max() - events[log_ids.start_time].min()).total_seconds()]
    return {
        "min": np.min(cycle_times),
        "q1": np.quantile(cycle_times, 0.25),
        "median": np.median(cycle_times),
        "mean": np.mean(cycle_times),
        "q3": np.quantile(cycle_times, 0.75),
        "max": np.max(cycle_times),
    }


def _simulate_and_evaluate(
    folder: Path,
    process: str,
    method: str,
    i: int,
    num_cases: int,
    start_timestamp: pd.Timestamp,
    test_log: pd.DataFrame,
) -> Tuple[float, float, dict]:
    # Simulate
    simulated_log_path = str(folder.joinpath("{}_sim_{}_{}.csv".format(process, method, i)))
    simulate(
        model_path=str(folder.joinpath("{}_{}.bpmn".format(process, method))),
        parameters_path=str(folder.joinpath("{}_{}.json".format(process, method))),
        num_cases=num_cases,
        starting_timestamp=start_timestamp,
        output_path=simulated_log_path,
    )
    # Read simulated log
    simulated_log = read_csv_log(simulated_log_path, event_log_ids)
    # Evaluate simulated log
    relative = relative_event_distribution_distance(test_log, event_log_ids, simulated_log, event_log_ids)
    absolute = absolute_event_distribution_distance(test_log, event_log_ids, simulated_log, event_log_ids)
    cycle_time_stats = compute_cycle_time_stats(simulated_log, event_log_ids)
    # Return measures
    return relative, absolute, cycle_time_stats


def export_sm(folder: Path, name: str, simulation_model: SimulationModel):
    simulation_model.bpmn_document.write(folder.joinpath(name + ".bpmn"), pretty_print=True)
    with open(folder.joinpath(name + ".json"), "w") as f:
        json.dump(simulation_model.simulation_parameters, f)


def _report_timers(folder: Path, name: str, enhancer: Union[DirectEnhancer, HyperOptEnhancer]):
    with open(folder.joinpath(name + "_timers.txt"), "w") as output_file:
        if type(enhancer) is DirectEnhancer:
            # Print timers
            for activity in enhancer.timers:
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
