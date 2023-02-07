import pandas as pd

from extraneous_activity_delays.config import Configuration
from extraneous_activity_delays.delay_discoverer import compute_naive_extraneous_activity_delays, \
    compute_complex_extraneous_activity_delays, _get_first_and_last_available
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
    assert len(delays_no_calendars) == 1


def test_compute_complex_extraneous_activity_delays():
    # Create Mon-Fri 9-17 calendars for DIO
    working_calendar = RCalendar("mon-fry-9-17")
    working_calendar.work_intervals[0] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[1] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[2] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[3] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[4] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    # Read event log with only resource off-duty waiting time
    event_log = read_csv_log("./tests/assets/event_log_1.csv", DEFAULT_CSV_IDS)
    config_with_calendars = Configuration(
        log_ids=DEFAULT_CSV_IDS, process_name="test_compute_complex_extraneous_activity_delays",
        max_alpha=5.0, num_iterations=10,
        working_schedules={'DIO': working_calendar}
    )
    # Compute delays
    delays_with_calendars = compute_complex_extraneous_activity_delays(event_log, config_with_calendars)
    # Assert there are no delays when using the calendars
    assert len(delays_with_calendars) == 0
    # Create no calendars configuration
    config_no_calendars = Configuration(
        log_ids=DEFAULT_CSV_IDS, process_name="test_compute_naive_extraneous_activity_delays",
        max_alpha=5.0, num_iterations=10
    )
    # Compute delays
    delays_no_calendars = compute_complex_extraneous_activity_delays(event_log, config_no_calendars)
    # Assert there are delays if not using calendars
    assert len(delays_no_calendars) == 1


def test__get_first_and_last_available():
    # Assert first and last are the beginning and end when empty
    beginning = pd.Timestamp("2023-01-25T09:00:00+00:00")
    end = pd.Timestamp("2023-01-25T09:00:00+00:00")
    assert _get_first_and_last_available(
        beginning=beginning,
        end=end,
        starts=[],
        ends=[],
        time_gap=pd.Timedelta(seconds=1)
    ) == (beginning, end)
    # Assert first and last are the beginning and end when empty
    beginning = pd.Timestamp("2023-01-25T09:00:00+00:00")
    end = pd.Timestamp("2023-01-25T10:00:00+00:00")
    assert _get_first_and_last_available(
        beginning=beginning,
        end=end,
        starts=[],
        ends=[],
        time_gap=pd.Timedelta(seconds=1)
    ) == (beginning, end)
    # Assert first is the end of an interval that overlaps with the start of the waiting period
    beginning = pd.Timestamp("2023-01-25T09:00:00+00:00")
    end = pd.Timestamp("2023-01-25T10:00:00+00:00")
    assert _get_first_and_last_available(
        beginning=beginning,
        end=end,
        starts=[pd.Timestamp("2023-01-25T08:00:00+00:00")],
        ends=[pd.Timestamp("2023-01-25T09:30:00+00:00")],
        time_gap=pd.Timedelta(seconds=1)
    ) == (pd.Timestamp("2023-01-25T09:30:00+00:00"), end)
    # Assert first is the end of an interval that overlaps with the start of the waiting period (with extra working element)
    beginning = pd.Timestamp("2023-01-25T09:00:00+00:00")
    end = pd.Timestamp("2023-01-25T10:00:00+00:00")
    assert _get_first_and_last_available(
        beginning=beginning,
        end=end,
        starts=[pd.Timestamp("2023-01-25T08:00:00+00:00"), pd.Timestamp("2023-01-25T09:45:00+00:00")],
        ends=[pd.Timestamp("2023-01-25T09:30:00+00:00"), pd.Timestamp("2023-01-25T09:49:00+00:00")],
        time_gap=pd.Timedelta(seconds=1)
    ) == (pd.Timestamp("2023-01-25T09:30:00+00:00"), end)
    # Assert last is the start of an interval that overlaps with the end of the waiting period
    beginning = pd.Timestamp("2023-01-25T09:00:00+00:00")
    end = pd.Timestamp("2023-01-25T10:00:00+00:00")
    assert _get_first_and_last_available(
        beginning=beginning,
        end=end,
        starts=[pd.Timestamp("2023-01-25T09:30:00+00:00")],
        ends=[pd.Timestamp("2023-01-25T11:00:00+00:00")],
        time_gap=pd.Timedelta(seconds=1)
    ) == (beginning, pd.Timestamp("2023-01-25T09:30:00+00:00"))
    # Assert first is the end of an interval that overlaps with the start of the waiting period (with extra working element)
    beginning = pd.Timestamp("2023-01-25T09:00:00+00:00")
    end = pd.Timestamp("2023-01-25T10:00:00+00:00")
    assert _get_first_and_last_available(
        beginning=beginning,
        end=end,
        starts=[pd.Timestamp("2023-01-25T09:09:00+00:00"), pd.Timestamp("2023-01-25T09:30:00+00:00")],
        ends=[pd.Timestamp("2023-01-25T09:21:00+00:00"), pd.Timestamp("2023-01-25T10:00:00+00:00")],
        time_gap=pd.Timedelta(seconds=1)
    ) == (beginning, pd.Timestamp("2023-01-25T09:30:00+00:00"))
    # Assert no free time when the gap is increased
    beginning = pd.Timestamp("2023-01-25T09:00:00+00:00")
    end = pd.Timestamp("2023-01-25T10:00:00+00:00")
    assert _get_first_and_last_available(
        beginning=beginning,
        end=end,
        starts=[pd.Timestamp("2023-01-25T09:09:00+00:00"), pd.Timestamp("2023-01-25T09:30:00+00:00")],
        ends=[pd.Timestamp("2023-01-25T09:21:00+00:00"), pd.Timestamp("2023-01-25T10:00:00+00:00")],
        time_gap=pd.Timedelta(minutes=10)
    ) == (end, end)
    # Assert when many activities in the middle, some breaking the time gap, some not
    beginning = pd.Timestamp("2023-01-25T09:00:00+00:00")
    end = pd.Timestamp("2023-01-25T12:00:00+00:00")
    assert _get_first_and_last_available(
        beginning=beginning,
        end=end,
        starts=[pd.Timestamp("2023-01-25T09:05:00+00:00"), pd.Timestamp("2023-01-25T09:30:00+00:00"),
                pd.Timestamp("2023-01-25T11:30:00+00:00"), pd.Timestamp("2023-01-25T11:55:00+00:00"),
                pd.Timestamp("2023-01-25T11:57:00+00:00")],
        ends=[pd.Timestamp("2023-01-25T09:15:00+00:00"), pd.Timestamp("2023-01-25T09:45:00+00:00"),
              pd.Timestamp("2023-01-25T11:35:00+00:00"), pd.Timestamp("2023-01-25T11:57:00+00:00"),
              pd.Timestamp("2023-01-25T11:59:00+00:00")],
        time_gap=pd.Timedelta(minutes=10)
    ) == (pd.Timestamp("2023-01-25T09:15:00+00:00"), pd.Timestamp("2023-01-25T11:55:00+00:00"))
