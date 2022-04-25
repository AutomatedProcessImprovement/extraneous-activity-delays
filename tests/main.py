import pandas as pd
from estimate_start_times.config import EventLogIDs
from lxml import etree

from extraneous_activity_delays.config import DEFAULT_CSV_IDS, Configuration
from extraneous_activity_delays.enhance_with_delays import enhance_bpmn_model_with_delays
from extraneous_activity_delays.metrics import absolute_hour_emd, trace_duration_emd
from extraneous_activity_delays.simulator import simulate_bpmn_model


def main(dataset: str, config: Configuration):
    # Log paths
    raw_log_path = str(config.PATH_INPUTS.joinpath(dataset + ".csv.gz"))
    raw_model_path = str(config.PATH_INPUTS.joinpath(dataset + ".bpmn"))
    enhanced_model_path = str(config.PATH_OUTPUTS.joinpath(dataset + ".bpmn"))
    # Read event log
    event_log = read_event_log(raw_log_path, config.log_ids)
    # Read BPMN model
    parser = etree.XMLParser(remove_blank_text=True)
    bpmn_document = etree.parse(raw_model_path, parser)
    # Enhance with activity delays
    enhance_bpmn_model_with_delays(event_log, bpmn_document, config)
    # Write enhanced simulation model
    bpmn_document.write(enhanced_model_path, pretty_print=True)
    # Simulate the enhanced BPMN
    raw_simulated_log_path = str(config.PATH_SIMULATED.joinpath(dataset + ".csv"))
    enhanced_simulated_log_path = str(config.PATH_SIMULATED.joinpath(dataset + "_enhanced.csv"))
    simulate_bpmn_model(raw_model_path, raw_simulated_log_path, config)
    simulate_bpmn_model(enhanced_model_path, enhanced_simulated_log_path, config)
    simulated_log_ids = EventLogIDs(
        case="caseid",
        activity="task",
        start_time="start_timestamp",
        end_time="end_timestamp",
        resource="resource"
    )
    # Compare the new log with the original one
    raw_simulated_log = read_event_log(raw_simulated_log_path, simulated_log_ids)
    enhanced_simulated_log = read_event_log(enhanced_simulated_log_path, simulated_log_ids)
    bin_size = max(
        [events[DEFAULT_CSV_IDS.end_time].max() - events[DEFAULT_CSV_IDS.start_time].min()
         for case, events in event_log.groupby([DEFAULT_CSV_IDS.case])]
    ) / 1000
    print(bin_size)
    print("{},{},{},{},{}".format(
        dataset,
        absolute_hour_emd(event_log, config.log_ids, raw_simulated_log, simulated_log_ids),
        absolute_hour_emd(event_log, config.log_ids, enhanced_simulated_log, simulated_log_ids),
        trace_duration_emd(event_log, config.log_ids, raw_simulated_log, simulated_log_ids, bin_size),
        trace_duration_emd(event_log, config.log_ids, enhanced_simulated_log, simulated_log_ids, bin_size)
    ))


def read_event_log(log_path: str, log_ids: EventLogIDs):
    event_log = pd.read_csv(log_path)
    event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time], utc=True)
    event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True)
    event_log[log_ids.resource].fillna("NOT_SET", inplace=True)
    event_log[log_ids.resource] = event_log[log_ids.resource].astype("string")
    return event_log


if __name__ == '__main__':
    # Print header
    print("dataset,timestamps_raw,timestamps_enhanced,cycle_time_raw,cycle_time_enhanced")
    # Configuration for raw logs
    configuration = Configuration(DEFAULT_CSV_IDS)
    # Launch analysis for each dataset
    main("Application_to_Approval_Government_Agency", configuration)
    main("BPI_Challenge_2012_W_Two_TS", configuration)
    main("BPI_Challenge_2017_W_Two_TS", configuration)
    main("callcentre", configuration)
    main("ConsultaDataMining201618", configuration)
    main("insurance", configuration)
    main("poc_processmining", configuration)
    main("Production", configuration)
