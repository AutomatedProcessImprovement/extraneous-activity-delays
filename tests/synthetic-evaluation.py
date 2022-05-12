import pandas as pd
from estimate_start_times.config import EventLogIDs
from lxml import etree

from extraneous_activity_delays.bpmn_enhancer import set_number_instances_to_simulate, set_start_datetime_to_simulate
from extraneous_activity_delays.config import Configuration
from extraneous_activity_delays.enhance_with_delays import HyperOptEnhancer

sim_log_ids = EventLogIDs(
    case="caseid",
    activity="task",
    start_time="start_timestamp",
    end_time="end_timestamp",
    resource="resource"
)


def experimentation_synthetic_logs():
    # Configuration
    config = Configuration(
        process_name="Pharmacy",
        instant_activities={" Check if refill is allowed", " Check DUR", " Check Insurance"},  # Pharmacy
        max_alpha=2.0,
        num_evaluations=100
    )
    # Read event log
    event_log = read_event_log("../inputs/synthetic-simulation-models/Pharmacy_train.csv.gz", config.log_ids)
    # Read BPMN model
    parser = etree.XMLParser(remove_blank_text=True)
    bpmn_document = etree.parse("../inputs/synthetic-simulation-models/Pharmacy.bpmn", parser)
    set_number_instances_to_simulate(bpmn_document, len(event_log[config.log_ids.case].unique()))
    set_start_datetime_to_simulate(bpmn_document, min(event_log[config.log_ids.start_time]))
    # Enhance with activity delays and hyper-optimization
    hyperopt_enhancer = HyperOptEnhancer(event_log, bpmn_document, config)
    hyperopt_enhanced_bpmn_document = hyperopt_enhancer.enhance_bpmn_model_with_delays()
    if len(hyperopt_enhancer.best_timers) == 0:
        print("No extraneous delays discovered!")
    for timer_activity in hyperopt_enhancer.best_timers:
        print(hyperopt_enhancer.best_timers[timer_activity])


def read_event_log(log_path: str, log_ids: EventLogIDs):
    event_log = pd.read_csv(log_path)
    event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time], utc=True)
    event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True)
    event_log[log_ids.resource].fillna("NOT_SET", inplace=True)
    event_log[log_ids.resource] = event_log[log_ids.resource].astype("string")
    return event_log


if __name__ == '__main__':
    experimentation_synthetic_logs()
