import pandas as pd
from estimate_start_times.config import EventLogIDs
from lxml import etree

from extraneous_activity_delays.bpmn_enhancer import enhance_bpmn_model_with_delays
from extraneous_activity_delays.config import DEFAULT_CSV_IDS
from extraneous_activity_delays.delay_discoverer import calculate_extraneous_activity_delays


def main(log_path: str, model_path: str, output_path: str, log_ids: EventLogIDs):
    # Read event log
    event_log = pd.read_csv(log_path)
    event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time], utc=True)
    event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True)
    event_log[log_ids.resource].fillna("NOT_SET", inplace=True)
    event_log[log_ids.resource] = event_log[log_ids.resource].astype("string")
    # Calculate extraneous delays
    timers = calculate_extraneous_activity_delays(event_log, log_ids)
    # Read BPMN model
    parser = etree.XMLParser(remove_blank_text=True)
    bpmn_document = etree.parse(model_path, parser)
    # Enhance process model
    enhance_bpmn_model_with_delays(bpmn_document, timers)
    # Write enhanced simulation model
    bpmn_document.write(output_path, pretty_print=True)


if __name__ == '__main__':
    main(
        "../event_logs/BPI_Challenge_2012_W_Two_TS.csv.gz",
        "../event_logs/BPI_Challenge_2012_W_Two_TS.bpmn",
        "../outputs/BPI_Challenge_2012_W_Two_TS_delays.bpmn",
        DEFAULT_CSV_IDS
    )
