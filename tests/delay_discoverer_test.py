import pandas as pd

from extraneous_activity_delays.config import Configuration
from extraneous_activity_delays.delay_discoverer import compute_naive_extraneous_activity_delays
from pix_utils.calendar.resource_calendar import RCalendar, Interval
from pix_utils.input import read_csv_log
from pix_utils.log_ids import DEFAULT_CSV_IDS


def test_compute_naive_extraneous_activity_delays():
    # Read event log with only resource off-duty waiting time
    event_log = read_csv_log("./tests/assets/event_log_1.csv", DEFAULT_CSV_IDS)
    # Create Mon-Fri 9-17 calendars for DIO
    working_calendar = RCalendar("mon-fry-9-17")
    working_calendar.work_intervals[0] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[1] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[2] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[3] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[4] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    config_with_calendars = Configuration(
        log_ids=DEFAULT_CSV_IDS, process_name="test_compute_naive_extraneous_activity_delays",
        max_alpha=5.0, num_iterations=10,
        working_schedules={'DIO': working_calendar}
    )
    # Compute delays
    delays_with_calendars = compute_naive_extraneous_activity_delays(event_log, config_with_calendars)
    # Assert there are no delays when using the calendars
    assert len(delays_with_calendars) == 0
    # Create no calendars configuration
    config_no_calendars = Configuration(
        log_ids=DEFAULT_CSV_IDS, process_name="test_compute_naive_extraneous_activity_delays",
        max_alpha=5.0, num_iterations=10
    )
    # Compute delays
    delays_no_calendars = compute_naive_extraneous_activity_delays(event_log, config_no_calendars)
    # Assert there are delays if not using calendars
    assert len(delays_no_calendars) > 0
