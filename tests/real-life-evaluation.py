from statistics import mean

import pandas as pd
from estimate_start_times.config import EventLogIDs
from lxml import etree

from extraneous_activity_delays.bpmn_enhancer import set_number_instances_to_simulate, set_start_datetime_to_simulate
from extraneous_activity_delays.config import Configuration
from extraneous_activity_delays.enhance_with_delays import HyperOptEnhancer, NaiveEnhancer
from extraneous_activity_delays.metrics import trace_duration_emd, absolute_hour_emd
from extraneous_activity_delays.simulator import simulate_bpmn_model
from extraneous_activity_delays.utils import create_folder

sim_log_ids = EventLogIDs(
    case="caseid",
    activity="task",
    start_time="start_timestamp",
    end_time="end_timestamp",
    resource="resource"
)


def experimentation_real_life():
    datasets = [
        ("BPIC_2012_W", "BPIC_2012_W_contained_Oct16_Dec15", "BPIC_2012_W_contained_Jan07_Mar08"),
        ("BPIC_2017_W", "BPIC_2017_W_contained_Jun20_Sep16", "BPIC_2017_W_contained_Sep17_Dec19"),
        ("Governmental_Agency", "Governmental_Agency_contained_Jul18-16_Jun30-17", "Governmental_Agency_contained_Jul03-17_Sep14-18"),
        ("poc_processmining", "poc_processmining_contained_Dec01_Feb15", "poc_processmining_contained_Feb16_Apr27")
    ]
    # Write CSV header
    with open("../outputs/real-life-evaluation/metrics.csv", 'a') as output_file:
        output_file.write("dataset,cycle_time_raw,cycle_time_naive_enhanced,cycle_time_hyperopt_enhanced,"
                          "cycle_time_hyperopt_enhanced_vs_train,timestamps_raw,timestamps_naive_enhanced,"
                          "timestamps_hyperopt_enhanced,timestamps_hyperopt_enhanced_vs_train\n")
    # Launch analysis for each dataset
    for dataset, train, test in datasets:
        with open("../outputs/real-life-evaluation/metrics.csv", 'a') as output_file:
            configuration = Configuration(process_name=dataset, max_alpha=2.0, num_evaluations=100)
            experimentation_real_life_run(dataset, train, test, configuration, output_file)


def experimentation_real_life_run(dataset: str, train_dataset: str, test_dataset: str, config: Configuration, metrics_file):
    # --- Raw paths --- #
    real_input_path = config.PATH_INPUTS.joinpath("real-life")
    train_log_path = str(real_input_path.joinpath(train_dataset + ".csv.gz"))
    test_log_path = str(real_input_path.joinpath(test_dataset + ".csv.gz"))
    original_model_path = str(real_input_path.joinpath(dataset + ".bpmn"))

    # --- Read event logs --- #
    train_log = read_event_log(train_log_path, config.log_ids)
    test_log = read_event_log(test_log_path, config.log_ids)

    # --- Read BPMN models --- #
    parser = etree.XMLParser(remove_blank_text=True)
    original_bpmn_model = etree.parse(original_model_path, parser)
    set_number_instances_to_simulate(original_bpmn_model, len(train_log[config.log_ids.case].unique()))
    set_start_datetime_to_simulate(original_bpmn_model, min(train_log[config.log_ids.start_time]))

    # --- Enhance with full discovered activity delays --- #
    naive_enhancer = NaiveEnhancer(train_log, original_bpmn_model, config)
    naive_enhanced_bpmn_model = naive_enhancer.enhance_bpmn_model_with_delays()

    # --- Enhance with hyper-parametrized activity delays --- #
    hyperopt_enhancer = HyperOptEnhancer(train_log, original_bpmn_model, config)
    hyperopt_enhanced_bpmn_model = hyperopt_enhancer.enhance_bpmn_model_with_delays()

    # --- Write BPMN models to files (change their start_time and num_instances to fit with test log) --- #
    evaluation_folder = config.PATH_OUTPUTS.joinpath("real-life-evaluation").joinpath(dataset)
    create_folder(evaluation_folder)
    # Original one
    set_number_instances_to_simulate(original_bpmn_model, len(test_log[config.log_ids.case].unique()))
    set_start_datetime_to_simulate(original_bpmn_model, min(test_log[config.log_ids.start_time]))
    original_bpmn_model_path = evaluation_folder.joinpath("{}_original.bpmn".format(dataset))
    original_bpmn_model.write(original_bpmn_model_path, pretty_print=True)
    # Enhanced with naive technique (full delays)
    set_number_instances_to_simulate(naive_enhanced_bpmn_model, len(test_log[config.log_ids.case].unique()))
    set_start_datetime_to_simulate(naive_enhanced_bpmn_model, min(test_log[config.log_ids.start_time]))
    naive_enhanced_bpmn_model_path = evaluation_folder.joinpath("{}_naive_enhanced.bpmn".format(dataset))
    naive_enhanced_bpmn_model.write(naive_enhanced_bpmn_model_path, pretty_print=True)
    # Enhanced with hyper-parametrized delays
    set_number_instances_to_simulate(hyperopt_enhanced_bpmn_model, len(test_log[config.log_ids.case].unique()))
    set_start_datetime_to_simulate(hyperopt_enhanced_bpmn_model, min(test_log[config.log_ids.start_time]))
    hyperopt_enhanced_bpmn_model_path = evaluation_folder.joinpath("{}_hyperopt_enhanced.bpmn".format(dataset))
    hyperopt_enhanced_bpmn_model.write(hyperopt_enhanced_bpmn_model_path, pretty_print=True)

    # --- Simulate and measure quality --- #
    bin_size = max(
        [events[config.log_ids.end_time].max() - events[config.log_ids.start_time].min()
         for case, events in test_log.groupby([config.log_ids.case])]
    ) / 1000  # 1.000 bins
    # Set lists to store the results of each comparison and get the mean
    original_cycle_emds, original_timestamps_emds = [], []
    naive_cycle_emds, naive_timestamps_emds = [], []
    hyperopt_cycle_emds, hyperopt_timestamps_emds = [], []
    hyperopt_vs_train_cycle_emds, hyperopt_vs_train_timestamps_emds = [], []
    # Simulate many times and compute the mean
    for i in range(config.num_evaluation_simulations):
        # Simulate, read, and evaluate original model
        original_simulated_log_path = str(evaluation_folder.joinpath("{}_sim_original_{}.csv".format(dataset, i)))
        simulate_bpmn_model(original_bpmn_model_path, original_simulated_log_path, config)
        original_simulated_event_log = read_event_log(original_simulated_log_path, sim_log_ids)
        original_cycle_emds += [trace_duration_emd(test_log, config.log_ids, original_simulated_event_log, sim_log_ids, bin_size)]
        original_timestamps_emds += [absolute_hour_emd(test_log, config.log_ids, original_simulated_event_log, sim_log_ids)]
        # Simulate, read, and evaluate naively enhanced model
        naive_simulated_log_path = str(evaluation_folder.joinpath("{}_sim_naive_enhanced_{}.csv".format(dataset, i)))
        simulate_bpmn_model(naive_enhanced_bpmn_model_path, naive_simulated_log_path, config)
        naive_simulated_event_log = read_event_log(naive_simulated_log_path, sim_log_ids)
        naive_cycle_emds += [trace_duration_emd(test_log, config.log_ids, naive_simulated_event_log, sim_log_ids, bin_size)]
        naive_timestamps_emds += [absolute_hour_emd(test_log, config.log_ids, naive_simulated_event_log, sim_log_ids)]
        # Simulate, read, and evaluate hyper-parametrized enhanced model (also against train)
        hyperopt_simulated_log_path = str(evaluation_folder.joinpath("{}_sim_hyperopt_enhanced_{}.csv".format(dataset, i)))
        simulate_bpmn_model(hyperopt_enhanced_bpmn_model_path, hyperopt_simulated_log_path, config)
        hyperopt_simulated_event_log = read_event_log(hyperopt_simulated_log_path, sim_log_ids)
        hyperopt_cycle_emds += [trace_duration_emd(test_log, config.log_ids, hyperopt_simulated_event_log, sim_log_ids, bin_size)]
        hyperopt_timestamps_emds += [absolute_hour_emd(test_log, config.log_ids, hyperopt_simulated_event_log, sim_log_ids)]
        hyperopt_vs_train_cycle_emds += [trace_duration_emd(train_log, config.log_ids, hyperopt_simulated_event_log, sim_log_ids, bin_size)]
        hyperopt_vs_train_timestamps_emds += [absolute_hour_emd(train_log, config.log_ids, hyperopt_simulated_event_log, sim_log_ids)]

    # --- Print results --- #
    metrics_file.write("{},{},{},{},{},{},{},{},{}\n".format(
        dataset,
        mean(original_cycle_emds),
        mean(naive_cycle_emds),
        mean(hyperopt_cycle_emds),
        mean(hyperopt_vs_train_cycle_emds),
        mean(original_timestamps_emds),
        mean(naive_timestamps_emds),
        mean(hyperopt_timestamps_emds),
        mean(hyperopt_vs_train_timestamps_emds)
    ))


def read_event_log(log_path: str, log_ids: EventLogIDs):
    event_log = pd.read_csv(log_path)
    event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time], utc=True)
    event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True)
    event_log[log_ids.resource].fillna("NOT_SET", inplace=True)
    event_log[log_ids.resource] = event_log[log_ids.resource].astype("string")
    return event_log


if __name__ == '__main__':
    experimentation_real_life()
