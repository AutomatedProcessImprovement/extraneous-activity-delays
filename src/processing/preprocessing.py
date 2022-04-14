import pandas as pd

from extraneous_activity_delays.config import DEFAULT_CSV_IDS

logs = [
    "insurance",
    "BPI_Challenge_2012_W_Two_TS",
    "BPI_Challenge_2017_W_Two_TS",
    "Application_to_Approval_Government_Agency",
    "callcentre",
    "ConsultaDataMining201618",
    "poc_processmining",
    "Production",
    "confidential",
    "Loan_Application",
    "cvs_pharmacy",
    "Procure_to_Pay",
]
raw_path = "../../event_logs/{}.csv.gz"
output_path = "../../event_logs/filtered/{}.csv.gz"


def preprocess_logs():
    log_ids = DEFAULT_CSV_IDS
    for log_name in logs:
        print(log_name)
        event_log = pd.read_csv(raw_path.format(log_name))
        filtered_event_log = event_log.drop_duplicates(
            subset=[log_ids.case, log_ids.activity, log_ids.start_time, log_ids.end_time, log_ids.resource]
        )
        print("\tRemoved {} rows out of {} ({:.2f}%)".format(
            len(event_log) - len(filtered_event_log),
            len(event_log),
            round((len(event_log) - len(filtered_event_log)) / len(event_log) * 100, 2)
        ))
        filtered_event_log.to_csv(output_path.format(log_name), encoding='utf-8', index=False, compression='gzip')
    print("\n\n")


if __name__ == '__main__':
    preprocess_logs()
