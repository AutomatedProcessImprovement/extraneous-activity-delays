from statistics import mean

import pandas as pd
from extraneous_activity_delays.config import Configuration, SimulationEngine, SimulationModel
from extraneous_activity_delays.enhance_with_delays import DirectEnhancer, HyperOptEnhancer
from extraneous_activity_delays.qbp.simulation_model_enhancer import (
    set_number_instances_to_simulate,
    set_start_datetime_to_simulate,
)
from extraneous_activity_delays.qbp.simulator import simulate
from extraneous_activity_delays.utils.file_manager import create_folder
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from lxml import etree
from pix_framework.io.event_log import EventLogIDs

sim_log_ids = EventLogIDs(
    case="caseid", activity="task", start_time="start_timestamp", end_time="end_timestamp", resource="resource"
)


def experimentation_synthetic_logs():
    datasets = [
        ("Pharmacy", "Pharmacy"),  # Format: tuple with first element
        ("Pharmacy_1_timer", "Pharmacy"),  # being the dataset to evaluate
        ("Pharmacy_4_timers", "Pharmacy"),  # and second element the dataset
        ("Loan_Application", "Loan_Application"),  # with no timers
        ("Loan_Application_1_timer", "Loan_Application"),
        ("Loan_Application_4_timers", "Loan_Application"),
        ("Procure_to_Pay", "Procure_to_Pay"),
        ("Procure_to_Pay_1_timer", "Procure_to_Pay"),
        ("Procure_to_Pay_4_timers", "Procure_to_Pay"),
        ("Insurance_Claims", "Insurance_Claims"),
        ("Insurance_Claims_1_timer", "Insurance_Claims"),
        ("Insurance_Claims_4_timers", "Insurance_Claims"),
        ("Confidential", "Confidential_noTimers"),
    ]
    # Write CSV header
    with open("../outputs/synthetic-evaluation/metrics.csv", "a") as output_file:
        output_file.write(
            "dataset,cycle_time_original,cycle_time_no_timers,cycle_time_direct_enhanced,"
            "cycle_time_hyperopt_enhanced,cycle_time_hyperopt_enhanced_vs_train,"
            "cycle_time_hyperopt_holdout_enhanced,timestamps_original,timestamps_no_timers,"
            "timestamps_direct_enhanced,timestamps_hyperopt_enhanced,"
            "timestamps_hyperopt_enhanced_vs_train,timestamps_hyperopt_holdout_enhanced\n"
        )
    # Launch analysis for each dataset
    for log, no_timers_log in datasets:
        with open("../outputs/synthetic-evaluation/metrics.csv", "a") as output_file:
            experimentation_synthetic_logs_run(log, no_timers_log, output_file)


def experimentation_synthetic_logs_run(dataset: str, no_timers_dataset: str, metrics_file):
    # Configuration
    config = Configuration(
        log_ids=sim_log_ids,
        process_name=dataset,
        instant_activities={"Check if refill is allowed", "Check DUR", "Check Insurance"},  # for Pharmacy log
        max_alpha=50.0,
        num_iterations=200,
        num_evaluation_simulations=10,
        simulation_engine=SimulationEngine.QBP,
    )
    hold_out_config = Configuration(
        log_ids=sim_log_ids,
        process_name=dataset,
        instant_activities={"Check if refill is allowed", "Check DUR", "Check Insurance"},  # for Pharmacy log
        max_alpha=50.0,
        training_partition_ratio=0.5,
        num_iterations=200,
        num_evaluation_simulations=10,
        simulation_engine=SimulationEngine.QBP,
    )

    # --- Raw paths --- #
    synthetic_input_path = config.PATH_INPUTS.joinpath("synthetic")
    train_log_path = str(synthetic_input_path.joinpath(dataset + "_train.csv.gz"))
    test_log_path = str(synthetic_input_path.joinpath(dataset + "_test.csv.gz"))
    original_model_path = str(synthetic_input_path.joinpath(dataset + ".bpmn"))
    no_timers_model_path = str(synthetic_input_path.joinpath(no_timers_dataset + ".bpmn"))

    # --- Evaluation folder --- #
    evaluation_folder = config.PATH_OUTPUTS.joinpath("synthetic-evaluation").joinpath(dataset)
    create_folder(evaluation_folder)

    # --- Read event logs --- #
    train_log = read_event_log(train_log_path, config.log_ids)
    test_log = read_event_log(test_log_path, config.log_ids)

    # --- Read BPMN models --- #
    parser = etree.XMLParser(remove_blank_text=True)
    original_bpmn_model = etree.parse(original_model_path, parser)
    no_timers_bpmn_model = etree.parse(no_timers_model_path, parser)
    set_number_instances_to_simulate(no_timers_bpmn_model, len(train_log[config.log_ids.case].unique()))
    set_start_datetime_to_simulate(no_timers_bpmn_model, min(train_log[config.log_ids.start_time]))
    simulation_model = SimulationModel(no_timers_bpmn_model)

    # --- Enhance with full discovered activity delays --- #
    direct_enhancer = DirectEnhancer(train_log, simulation_model, config)
    direct_enhanced_bpmn_model = direct_enhancer.enhance_simulation_model_with_delays().bpmn_document
    with open(evaluation_folder.joinpath("direct_enhancer_timers.txt"), "w") as output_file:
        for activity in direct_enhancer.timers:
            output_file.write("'{}': {}\n".format(activity, direct_enhancer.timers[activity]))

    # --- Enhance with hyper-parametrized activity delays --- #
    hyperopt_enhancer = HyperOptEnhancer(train_log, simulation_model, config)
    hyperopt_enhanced_bpmn_model = hyperopt_enhancer.enhance_simulation_model_with_delays().bpmn_document
    with open(evaluation_folder.joinpath("hyperopt_enhancer_timers.txt"), "w") as output_file:
        for activity in hyperopt_enhancer.best_timers:
            output_file.write("'{}': {}\n".format(activity, hyperopt_enhancer.best_timers[activity]))
        output_file.write("\nLosses: {}".format(hyperopt_enhancer.losses))

    # --- Enhance with hyper-parametrized activity delays with hold-out --- #
    hyperopt_holdout_enhancer = HyperOptEnhancer(train_log, simulation_model, hold_out_config)
    hyperopt_holdout_enhanced_bpmn_model = (
        hyperopt_holdout_enhancer.enhance_simulation_model_with_delays().bpmn_document
    )
    with open(evaluation_folder.joinpath("hyperopt_enhancer_timers_holdout.txt"), "w") as output_file:
        for activity in hyperopt_holdout_enhancer.best_timers:
            output_file.write("'{}': {}\n".format(activity, hyperopt_holdout_enhancer.best_timers[activity]))
        output_file.write("\nLosses: {}".format(hyperopt_holdout_enhancer.losses))

    # --- Write BPMN models to files (change their start_time and num_instances to fit with test log) --- #
    # Original one (with or without timers, it depends on the test)
    set_number_instances_to_simulate(original_bpmn_model, len(test_log[config.log_ids.case].unique()))
    set_start_datetime_to_simulate(original_bpmn_model, min(test_log[config.log_ids.start_time]))
    original_bpmn_model_path = evaluation_folder.joinpath("{}_original.bpmn".format(dataset))
    original_bpmn_model.write(original_bpmn_model_path, pretty_print=True)
    # Original without timers, the one we enhanced
    set_number_instances_to_simulate(no_timers_bpmn_model, len(test_log[config.log_ids.case].unique()))
    set_start_datetime_to_simulate(no_timers_bpmn_model, min(test_log[config.log_ids.start_time]))
    no_timers_bpmn_model_path = evaluation_folder.joinpath("{}_no_timers.bpmn".format(dataset))
    no_timers_bpmn_model.write(no_timers_bpmn_model_path, pretty_print=True)
    # Enhanced with direct technique (full delays)
    set_number_instances_to_simulate(direct_enhanced_bpmn_model, len(test_log[config.log_ids.case].unique()))
    set_start_datetime_to_simulate(direct_enhanced_bpmn_model, min(test_log[config.log_ids.start_time]))
    direct_enhanced_bpmn_model_path = evaluation_folder.joinpath("{}_direct_enhanced.bpmn".format(dataset))
    direct_enhanced_bpmn_model.write(direct_enhanced_bpmn_model_path, pretty_print=True)
    # Enhanced with hyper-parametrized delays
    set_number_instances_to_simulate(hyperopt_enhanced_bpmn_model, len(test_log[config.log_ids.case].unique()))
    set_start_datetime_to_simulate(hyperopt_enhanced_bpmn_model, min(test_log[config.log_ids.start_time]))
    hyperopt_enhanced_bpmn_model_path = evaluation_folder.joinpath("{}_hyperopt_enhanced.bpmn".format(dataset))
    hyperopt_enhanced_bpmn_model.write(hyperopt_enhanced_bpmn_model_path, pretty_print=True)
    # Enhanced with hyper-parametrized (with hold-out) delays
    set_number_instances_to_simulate(hyperopt_holdout_enhanced_bpmn_model, len(test_log[config.log_ids.case].unique()))
    set_start_datetime_to_simulate(hyperopt_holdout_enhanced_bpmn_model, min(test_log[config.log_ids.start_time]))
    hyperopt_holdout_enhanced_bpmn_model_path = evaluation_folder.joinpath(
        "{}_hyperopt_holdout_enhanced.bpmn".format(dataset)
    )
    hyperopt_holdout_enhanced_bpmn_model.write(hyperopt_holdout_enhanced_bpmn_model_path, pretty_print=True)

    # --- Simulate and measure quality --- #
    bin_size = (
        max(
            [
                events[config.log_ids.end_time].max() - events[config.log_ids.start_time].min()
                for case, events in test_log.groupby([config.log_ids.case])
            ]
        )
        / 1000
    )  # 1.000 bins
    # Set lists to store the results of each comparison and get the mean
    original_cycle_emds, original_timestamps_emds = [], []
    no_timers_cycle_emds, no_timers_timestamps_emds = [], []
    direct_cycle_emds, direct_timestamps_emds = [], []
    hyperopt_cycle_emds, hyperopt_timestamps_emds = [], []
    hyperopt_vs_train_cycle_emds, hyperopt_vs_train_timestamps_emds = [], []
    hyperopt_holdout_cycle_emds, hyperopt_holdout_timestamps_emds = [], []
    # Simulate many times and compute the mean
    for i in range(config.num_evaluation_simulations):
        # Simulate, read, and evaluate original model
        original_simulated_log_path = str(evaluation_folder.joinpath("{}_sim_original_{}.csv".format(dataset, i)))
        simulate(original_bpmn_model_path, original_simulated_log_path, config)
        original_simulated_event_log = read_event_log(original_simulated_log_path, sim_log_ids)
        original_cycle_emds += [
            cycle_time_distribution_distance(
                test_log, config.log_ids, original_simulated_event_log, sim_log_ids, bin_size
            )
        ]
        original_timestamps_emds += [
            absolute_event_distribution_distance(test_log, config.log_ids, original_simulated_event_log, sim_log_ids)
        ]
        # Simulate, read, and evaluate original model with no timers
        no_timers_simulated_log_path = str(evaluation_folder.joinpath("{}_sim_no_timers_{}.csv".format(dataset, i)))
        simulate(no_timers_bpmn_model_path, no_timers_simulated_log_path, config)
        no_timers_simulated_event_log = read_event_log(no_timers_simulated_log_path, sim_log_ids)
        no_timers_cycle_emds += [
            cycle_time_distribution_distance(
                test_log, config.log_ids, no_timers_simulated_event_log, sim_log_ids, bin_size
            )
        ]
        no_timers_timestamps_emds += [
            absolute_event_distribution_distance(test_log, config.log_ids, no_timers_simulated_event_log, sim_log_ids)
        ]
        # Simulate, read, and evaluate directly enhanced model
        direct_simulated_log_path = str(evaluation_folder.joinpath("{}_sim_direct_enhanced_{}.csv".format(dataset, i)))
        simulate(direct_enhanced_bpmn_model_path, direct_simulated_log_path, config)
        direct_simulated_event_log = read_event_log(direct_simulated_log_path, sim_log_ids)
        direct_cycle_emds += [
            cycle_time_distribution_distance(
                test_log, config.log_ids, direct_simulated_event_log, sim_log_ids, bin_size
            )
        ]
        direct_timestamps_emds += [
            absolute_event_distribution_distance(test_log, config.log_ids, direct_simulated_event_log, sim_log_ids)
        ]
        # Simulate, read, and evaluate hyper-parametrized enhanced model (also against train)
        hyperopt_simulated_log_path = str(
            evaluation_folder.joinpath("{}_sim_hyperopt_enhanced_{}.csv".format(dataset, i))
        )
        simulate(hyperopt_enhanced_bpmn_model_path, hyperopt_simulated_log_path, config)
        hyperopt_simulated_event_log = read_event_log(hyperopt_simulated_log_path, sim_log_ids)
        hyperopt_cycle_emds += [
            cycle_time_distribution_distance(
                test_log, config.log_ids, hyperopt_simulated_event_log, sim_log_ids, bin_size
            )
        ]
        hyperopt_timestamps_emds += [
            absolute_event_distribution_distance(test_log, config.log_ids, hyperopt_simulated_event_log, sim_log_ids)
        ]
        displaced_hyperopt = hyperopt_simulated_event_log.copy()
        start_time_difference = (
            displaced_hyperopt[sim_log_ids.start_time].min() - train_log[config.log_ids.start_time].min()
        )
        displaced_hyperopt[sim_log_ids.start_time] = displaced_hyperopt[sim_log_ids.start_time] - start_time_difference
        displaced_hyperopt[sim_log_ids.end_time] = displaced_hyperopt[sim_log_ids.end_time] - start_time_difference
        hyperopt_vs_train_cycle_emds += [
            cycle_time_distribution_distance(train_log, config.log_ids, displaced_hyperopt, sim_log_ids, bin_size)
        ]
        hyperopt_vs_train_timestamps_emds += [
            absolute_event_distribution_distance(train_log, config.log_ids, displaced_hyperopt, sim_log_ids)
        ]
        # Simulate, read, and evaluate hyper-parametrized (with hold-out) enhanced model (also against train)
        hyperopt_holdout_simulated_log_path = str(
            evaluation_folder.joinpath("{}_sim_hyperopt_holdout_enhanced_{}.csv".format(dataset, i))
        )
        simulate(hyperopt_holdout_enhanced_bpmn_model_path, hyperopt_holdout_simulated_log_path, config)
        hyperopt_holdout_simulated_event_log = read_event_log(hyperopt_holdout_simulated_log_path, sim_log_ids)
        hyperopt_holdout_cycle_emds += [
            cycle_time_distribution_distance(
                test_log, config.log_ids, hyperopt_holdout_simulated_event_log, sim_log_ids, bin_size
            )
        ]
        hyperopt_holdout_timestamps_emds += [
            absolute_event_distribution_distance(
                test_log, config.log_ids, hyperopt_holdout_simulated_event_log, sim_log_ids
            )
        ]

    # --- Print results --- #
    metrics_file.write(
        "{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            dataset,
            mean(original_cycle_emds),
            mean(no_timers_cycle_emds),
            mean(direct_cycle_emds),
            mean(hyperopt_cycle_emds),
            mean(hyperopt_vs_train_cycle_emds),
            mean(hyperopt_holdout_cycle_emds),
            mean(original_timestamps_emds),
            mean(no_timers_timestamps_emds),
            mean(direct_timestamps_emds),
            mean(hyperopt_timestamps_emds),
            mean(hyperopt_vs_train_timestamps_emds),
            mean(hyperopt_holdout_timestamps_emds),
        )
    )


def read_event_log(log_path: str, log_ids: EventLogIDs):
    event_log = pd.read_csv(log_path)
    event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time], utc=True)
    event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True)
    event_log[log_ids.resource].fillna("NOT_SET", inplace=True)
    event_log[log_ids.resource] = event_log[log_ids.resource].astype("string")
    return event_log


if __name__ == "__main__":
    experimentation_synthetic_logs()
