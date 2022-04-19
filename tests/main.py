import pandas as pd

from extraneous_activity_delays.config import DEFAULT_CSV_IDS
from extraneous_activity_delays.delay_discoverer import calculate_extraneous_activity_delays

if __name__ == '__main__':
    log_ids = DEFAULT_CSV_IDS
    # Read event log
    event_log = pd.read_csv("../event_logs/BPI_Challenge_2012_W_Two_TS.csv.gz")
    event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time], utc=True)
    event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True)
    event_log[log_ids.resource].fillna("NOT_SET", inplace=True)
    event_log[log_ids.resource] = event_log[log_ids.resource].astype("string")
    # Calculate extraneous delays
    calculate_extraneous_activity_delays(event_log, log_ids)
