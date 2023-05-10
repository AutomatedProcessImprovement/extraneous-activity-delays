import os

import pandas as pd

from extraneous_activity_delays.prosimos.simulator import simulate
from pix_framework.log_ids import EventLogIDs


def transform_delays_into_wt():
    logs = [
        "Insurance_Claims", "Insurance_Claims_1_timer", "Insurance_Claims_3_timers", "Insurance_Claims_5_timers",
        "Loan_Application", "Loan_Application_1_timer", "Loan_Application_3_timers", "Loan_Application_5_timers",
        "Pharmacy", "Pharmacy_1_timer", "Pharmacy_3_timers", "Pharmacy_5_timers",
        "Procure_to_Pay", "Procure_to_Pay_1_timer", "Procure_to_Pay_3_timers", "Procure_to_Pay_5_timers"
    ]
    log_ids = EventLogIDs(case="case_id", enabled_time="enable_time", start_time="start_time", end_time="end_time")
    events = {
        "Event_0uhfx88": "Analyze Purchase Requisition",
        "Event_12h1qe9": "Analyze Request for Quotation",
        "Event_1lcb2pb": "Confirm Purchase Order",
        "Event_0sk0f7u": "Send invoice",
        "Event_0t470fr": "Pay invoice",
        "Event_0f8q4fm": "Enter prescription details",
        "Event_0tsn05q": "Check DUR",
        "Event_020svte": "Check Insurance",
        "Event_1hrnfk3": "Pack the drugs (Production)",
        "Event_0jj7p9e": "Pick-up",
        "Event_19s1gwa": "Applicant completes form",
        "Event_1g228xj": "Assess loan risk",
        "Event_0m4nhkw": "Reject application",
        "Event_0ygf1qj": "Approve application",
        "Event_18md465": "Cancel application",
        "Event_1nj1nea": "Close Claim",
        "Event_1f2tnvq": "Advise Claimant on Reimbursement",
        "Event_118uxuk": "Initiate Payment",
        "Event_09vssya": "Assess Claim",
        "Event_1p6n52e": "Determine likelihood of the claim",
    }
    for process_name in logs:
        model_path = "../inputs/synthetic/{}.bpmn".format(process_name)
        parameters_path = "../inputs/synthetic/{}.json".format(process_name)
        output_path = "../inputs/synthetic/{}_raw.csv".format(process_name)
        preprocessed_path = "../inputs/synthetic/{}.csv".format(process_name)
        # Simulate with prosimos
        simulate(model_path, parameters_path, 1000, pd.Timestamp("01/02/2023 09:00:00+00:00"), output_path, True)
        # Read simulated log
        event_log = pd.read_csv(output_path)
        event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time])
        event_log[log_ids.start_time] = event_log.apply(_remove_microsecond_start, axis=1)
        event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time])
        event_log[log_ids.end_time] = event_log.apply(_remove_microsecond_end, axis=1)
        event_log['extraneous_delay'] = 0
        event_log.drop(log_ids.enabled_time, axis=1, inplace=True)
        # Associate the timer delay as WT to the activity
        indexes, values = [], []
        for index, event in event_log[event_log[log_ids.activity].isin(list(events.keys()))].iterrows():
            next_event = event_log[
                (event_log[log_ids.case] == event[log_ids.case]) &
                (event_log[log_ids.start_time] >= event[log_ids.end_time]) &
                (event_log[log_ids.activity] == events[event[log_ids.activity]])
                ].sort_values(log_ids.start_time).iloc[0]
            indexes += [next_event.name]
            values += [(event[log_ids.end_time] - event[log_ids.start_time]).total_seconds()]
        event_log.loc[indexes, 'extraneous_delay'] = values
        # Retain only activity instances (no events)
        event_log = event_log[~event_log[log_ids.activity].isin(list(events.keys()))]
        # Write to file
        event_log.to_csv(preprocessed_path, index=False)
        os.remove(output_path)


def generate_single_logs():
    logs = [
        # "Confidential",
        "Insurance_Claims_5_timers", "Loan_Application_5_timers",
        "Pharmacy_5_timers", "Procure_to_Pay_5_timers"
    ]
    log_ids = EventLogIDs(case="case_id", enabled_time="enable_time", start_time="start_time", end_time="end_time")
    for process_name in logs:
        # model_path = "../inputs/synthetic/{}.bpmn".format(process_name)
        model_path = "../inputs/synthetic/{}_after.bpmn".format(process_name)
        parameters_path = "../inputs/synthetic/{}.json".format(process_name)
        # output_path = "../inputs/synthetic/{}.csv".format(process_name)
        output_path = "../inputs/synthetic/{}_after.csv".format(process_name)
        # Simulate with prosimos
        simulate(model_path, parameters_path, 1000, pd.Timestamp("01/02/2023 09:00:00+00:00"), output_path, False)
        # Read simulated log
        event_log = pd.read_csv(output_path)
        event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time])
        event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time])
        event_log.drop(log_ids.enabled_time, axis=1, inplace=True)
        os.remove(output_path)
        event_log.to_csv(output_path, index=False)


def _remove_microsecond_start(row):
    return row['start_time'].round('10L')


def _remove_microsecond_end(row):
    return row['end_time'].round('10L')


if __name__ == '__main__':
    transform_delays_into_wt()
