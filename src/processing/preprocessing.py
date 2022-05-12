import os

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
raw_path = "../../inputs/{}.csv.gz"
output_path = "../../inputs/filtered/{}.csv.gz"


def remove_duplicated_events():
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


def remove_intermediate_event_instances():
    timer_ids = [
        ' EVENT 27 CATCH TIMER', ' EVENT 28 CATCH TIMER', ' EVENT 29 CATCH TIMER', ' EVENT 30 CATCH TIMER', ' EVENT 31 CATCH TIMER',
        ' EVENT 32 CATCH TIMER', ' EVENT 33 CATCH TIMER', ' EVENT 34 CATCH TIMER', ' EVENT 35 CATCH TIMER', ' EVENT 36 CATCH TIMER',
        ' EVENT 37 CATCH TIMER', ' EVENT 3 START', ' EVENT 27 END', ' EVENT 18 END ERROR', " Loan application rejected",
        " Loan  application approved", " Loan  application canceled", " Loan  application received", " Prescription received",
        " Prescription fulfilled"
    ]
    for log in [
        "Loan_Application_1_timer_train.csv.gz", "Loan_Application_1_timer_test.csv.gz",
        "Loan_Application_4_timers_train.csv.gz", "Loan_Application_4_timers_test.csv.gz",
        "Loan_Application_train.csv.gz", "Loan_Application_test.csv.gz",
        "Pharmacy_1_timer_train.csv.gz", "Pharmacy_1_timer_test.csv.gz",
        "Pharmacy_4_timers_train.csv.gz", "Pharmacy_4_timers_test.csv.gz",
        "Pharmacy_train.csv.gz", "Pharmacy_test.csv.gz",
        "Procure_to_Pay_1_timer_train.csv.gz", "Procure_to_Pay_1_timer_test.csv.gz",
        "Procure_to_Pay_4_timers_train.csv.gz", "Procure_to_Pay_4_timers_test.csv.gz",
        "Procure_to_Pay_train.csv.gz", "Procure_to_Pay_test.csv.gz"
    ]:
        print("\n" + log)
        log_path = "../../inputs/synthetic-simulation-models/" + log
        event_log = pd.read_csv(log_path)
        event_log = event_log[~event_log['Activity'].isin(timer_ids)]
        os.remove(log_path)
        event_log.to_csv(log_path, index=False, encoding="utf-8")


if __name__ == '__main__':
    remove_intermediate_event_instances()
