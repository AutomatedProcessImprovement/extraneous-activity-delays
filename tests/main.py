from statistics import mean

import pandas as pd
from estimate_start_times.config import EventLogIDs
from lxml import etree

from extraneous_activity_delays.bpmn_enhancer import set_number_instances_to_simulate
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


def main(dataset: str, config: Configuration, metrics_file):
    # Raw paths
    raw_log_path = str(config.PATH_INPUTS.joinpath(dataset + ".csv.gz"))
    raw_model_path = str(config.PATH_INPUTS.joinpath(dataset + ".bpmn"))
    # Read event log
    event_log = read_event_log(raw_log_path, config.log_ids)
    # Read BPMN model
    parser = etree.XMLParser(remove_blank_text=True)
    bpmn_document = etree.parse(raw_model_path, parser)
    set_number_instances_to_simulate(bpmn_document, len(event_log[config.log_ids.case].unique()))
    # Enhance with activity delays
    naive_enhancer = NaiveEnhancer(event_log, bpmn_document, config)
    naive_enhanced_bpmn_document = naive_enhancer.enhance_bpmn_model_with_delays()
    # Enhance with activity delays and hyper-optimization
    hyperopt_enhancer = HyperOptEnhancer(event_log, bpmn_document, config)
    hyperopt_enhanced_bpmn_document = hyperopt_enhancer.enhance_bpmn_model_with_delays()
    # Write BPMN models to files
    evaluation_folder = config.PATH_OUTPUTS.joinpath("evaluation").joinpath(dataset)
    create_folder(evaluation_folder)
    raw_bpmn_model_path = evaluation_folder.joinpath("{}_raw.bpmn".format(dataset))
    bpmn_document.write(raw_bpmn_model_path, pretty_print=True)
    naive_enhanced_bpmn_model_path = evaluation_folder.joinpath("{}_naive_enhanced.bpmn".format(dataset))
    naive_enhanced_bpmn_document.write(naive_enhanced_bpmn_model_path, pretty_print=True)
    hyperopt_enhanced_bpmn_model_path = evaluation_folder.joinpath("{}_hyperopt_enhanced.bpmn".format(dataset))
    hyperopt_enhanced_bpmn_document.write(hyperopt_enhanced_bpmn_model_path, pretty_print=True)
    # Simulate and measure quality
    bin_size = max(
        [events[config.log_ids.end_time].max() - events[config.log_ids.start_time].min()
         for case, events in event_log.groupby([config.log_ids.case])]
    ) / 1000
    raw_cycle_emds = []
    naive_cycle_emds = []
    hyperopt_cycle_emds = []
    raw_timestamps_emds = []
    naive_timestamps_emds = []
    hyperopt_timestamps_emds = []
    for i in range(config.num_evaluation_simulations):
        # Simulate with model
        raw_simulated_log_path = str(evaluation_folder.joinpath("{}_sim_raw_{}.csv".format(dataset, i)))
        simulate_bpmn_model(raw_bpmn_model_path, raw_simulated_log_path, config)
        naive_simulated_log_path = str(evaluation_folder.joinpath("{}_sim_naive_enhanced_{}.csv".format(dataset, i)))
        simulate_bpmn_model(naive_enhanced_bpmn_model_path, naive_simulated_log_path, config)
        hyperopt_simulated_log_path = str(evaluation_folder.joinpath("{}_sim_hyperopt_enhanced_{}.csv".format(dataset, i)))
        simulate_bpmn_model(hyperopt_enhanced_bpmn_model_path, hyperopt_simulated_log_path, config)
        # Read simulated event logs
        raw_simulated_event_log = read_event_log(raw_simulated_log_path, sim_log_ids)
        naive_simulated_event_log = read_event_log(naive_simulated_log_path, sim_log_ids)
        hyperopt_simulated_event_log = read_event_log(hyperopt_simulated_log_path, sim_log_ids)
        # Measure log distances
        raw_cycle_emds += [trace_duration_emd(event_log, config.log_ids, raw_simulated_event_log, sim_log_ids, bin_size)]
        naive_cycle_emds += [trace_duration_emd(event_log, config.log_ids, naive_simulated_event_log, sim_log_ids, bin_size)]
        hyperopt_cycle_emds += [trace_duration_emd(event_log, config.log_ids, hyperopt_simulated_event_log, sim_log_ids, bin_size)]
        raw_timestamps_emds += [absolute_hour_emd(event_log, config.log_ids, raw_simulated_event_log, sim_log_ids)]
        naive_timestamps_emds += [absolute_hour_emd(event_log, config.log_ids, naive_simulated_event_log, sim_log_ids)]
        hyperopt_timestamps_emds += [absolute_hour_emd(event_log, config.log_ids, hyperopt_simulated_event_log, sim_log_ids)]
    # Print results
    metrics_file.write("{},{},{},{},{},{},{}\n".format(
        dataset,
        mean(raw_cycle_emds),
        mean(naive_cycle_emds),
        mean(hyperopt_cycle_emds),
        mean(raw_timestamps_emds),
        mean(naive_timestamps_emds),
        mean(hyperopt_timestamps_emds)
    ))


def read_event_log(log_path: str, log_ids: EventLogIDs):
    event_log = pd.read_csv(log_path)
    event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time], utc=True)
    event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True)
    event_log[log_ids.resource].fillna("NOT_SET", inplace=True)
    event_log[log_ids.resource] = event_log[log_ids.resource].astype("string")
    return event_log


if __name__ == '__main__':
    datasets = [
        "Application_to_Approval_Government_Agency",
        "BPI_Challenge_2012_W_Two_TS",
        "BPI_Challenge_2017_W_Two_TS",
        "callcentre",
        "ConsultaDataMining201618",
        "insurance",
        "poc_processmining",
        "Production"
    ]
    with open("../outputs/evaluation/metrics.csv", 'a') as output_file:
        output_file.write("dataset,cycle_time_raw,cycle_time_naive_enhanced,cycle_time_hyperopt_enhanced,"
                          "timestamps_raw,timestamps_naive_enhanced,timestamps_hyperopt_enhanced\n")
        # Launch analysis for each dataset
        for dataset in datasets:
            configuration = Configuration(process_name=dataset, num_evaluations=100)
            main(dataset, configuration, output_file)
