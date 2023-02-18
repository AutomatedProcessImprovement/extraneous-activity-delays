import pandas as pd

from extraneous_activity_delays.config import Configuration, TimerPlacement
from extraneous_activity_delays.delay_discoverer import compute_naive_extraneous_activity_delays, \
    compute_complex_extraneous_activity_delays, _get_first_and_last_available
from pix_utils.calendar.resource_calendar import RCalendar, Interval
from pix_utils.input import read_csv_log
from pix_utils.log_ids import DEFAULT_CSV_IDS


def test_compute_naive_extraneous_activity_delays():
    # Create Mon-Fri 9-17 calendars for DIO
    working_calendar = RCalendar("mon-fry-9-17")
    working_calendar.work_intervals[0] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[1] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[2] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[3] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[4] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]

    # Discover extraneous delays (BEFORE) without availability calendars
    event_log = read_csv_log("./tests/assets/event_log_1.csv", DEFAULT_CSV_IDS)
    config_no_calendars = Configuration(timer_placement=TimerPlacement.BEFORE, log_ids=DEFAULT_CSV_IDS)
    delays_no_calendars = compute_naive_extraneous_activity_delays(event_log, config_no_calendars)
    # Assert D has a delay if not using calendars
    assert 'D' in delays_no_calendars
    assert len(delays_no_calendars) == 1

    # Discover extraneous delays (BEFORE) with availability calendars
    event_log = read_csv_log("./tests/assets/event_log_1.csv", DEFAULT_CSV_IDS)
    config_with_calendars = Configuration(timer_placement=TimerPlacement.BEFORE, log_ids=DEFAULT_CSV_IDS,
                                          working_schedules={'DIO': working_calendar})
    delays_with_calendars = compute_naive_extraneous_activity_delays(event_log, config_with_calendars)
    # Assert there are no delays when using the calendars
    assert len(delays_with_calendars) == 0

    # Discover extraneous delays (AFTER) without availability calendars
    event_log = read_csv_log("./tests/assets/event_log_1.csv", DEFAULT_CSV_IDS)
    config_no_calendars = Configuration(timer_placement=TimerPlacement.AFTER, log_ids=DEFAULT_CSV_IDS)
    delays_no_calendars = compute_naive_extraneous_activity_delays(event_log, config_no_calendars)
    # Assert C has a delay if not using calendars
    assert 'C' in delays_no_calendars
    assert len(delays_no_calendars) == 1

    # Discover extraneous delays (BEFORE) with availability calendars
    event_log = read_csv_log("./tests/assets/event_log_1.csv", DEFAULT_CSV_IDS)
    config_with_calendars = Configuration(timer_placement=TimerPlacement.AFTER, log_ids=DEFAULT_CSV_IDS,
                                          working_schedules={'DIO': working_calendar})
    delays_with_calendars = compute_naive_extraneous_activity_delays(event_log, config_with_calendars)
    # Assert there are no delays when using the calendars
    assert len(delays_with_calendars) == 0


def test_compute_complex_extraneous_activity_delays():
    # Create Mon-Fri 9-17 calendars for DIO
    working_calendar = RCalendar("mon-fry-9-17")
    working_calendar.work_intervals[0] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[1] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[2] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[3] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]
    working_calendar.work_intervals[4] = [Interval(pd.Timestamp("2023-01-25T09:00:00+00:00"), pd.Timestamp("2023-01-25T17:00:00+00:00"))]

    # Discover extraneous delays (BEFORE) without availability calendars
    event_log = read_csv_log("./tests/assets/event_log_1.csv", DEFAULT_CSV_IDS)
    config_no_calendars = Configuration(timer_placement=TimerPlacement.BEFORE, log_ids=DEFAULT_CSV_IDS)
    delays_no_calendars = compute_complex_extraneous_activity_delays(event_log, config_no_calendars)
    # Assert there are delays if not using calendars
    assert 'D' in delays_no_calendars
    assert len(delays_no_calendars) == 1

    # Discover extraneous delays (BEFORE) with availability calendars
    event_log = read_csv_log("./tests/assets/event_log_1.csv", DEFAULT_CSV_IDS)
    config_with_calendars = Configuration(timer_placement=TimerPlacement.BEFORE, log_ids=DEFAULT_CSV_IDS,
                                          working_schedules={'DIO': working_calendar})
    delays_with_calendars = compute_complex_extraneous_activity_delays(event_log, config_with_calendars)
    # Assert there are no delays when using the calendars
    assert len(delays_with_calendars) == 0

    # Discover extraneous delays (AFTER) without availability calendars
    event_log = read_csv_log("./tests/assets/event_log_1.csv", DEFAULT_CSV_IDS)
    config_no_calendars = Configuration(timer_placement=TimerPlacement.AFTER, log_ids=DEFAULT_CSV_IDS)
    delays_no_calendars = compute_complex_extraneous_activity_delays(event_log, config_no_calendars)
    # Assert there are delays if not using calendars
    assert 'C' in delays_no_calendars
    assert len(delays_no_calendars) == 1

    # Discover extraneous delays (AFTER) with availability calendars
    event_log = read_csv_log("./tests/assets/event_log_1.csv", DEFAULT_CSV_IDS)
    config_with_calendars = Configuration(timer_placement=TimerPlacement.AFTER, log_ids=DEFAULT_CSV_IDS,
                                          working_schedules={'DIO': working_calendar})
    delays_with_calendars = compute_complex_extraneous_activity_delays(event_log, config_with_calendars)
    # Assert there are no delays when using the calendars
    assert len(delays_with_calendars) == 0


def test_compute_naive_extraneous_activity_delays_LoanApp():
    # Discover extraneous delays (BEFORE) without availability calendars
    event_log = read_csv_log("./tests/assets/LoanApp_no_delays.csv", DEFAULT_CSV_IDS)
    config = Configuration(timer_placement=TimerPlacement.BEFORE, log_ids=DEFAULT_CSV_IDS)
    delays = compute_naive_extraneous_activity_delays(event_log, config)
    # Assert there are delays if not using calendars
    assert len(delays) == 0

    # Discover extraneous delays (AFTER) without availability calendars
    event_log = read_csv_log("./tests/assets/LoanApp_no_delays.csv", DEFAULT_CSV_IDS)
    config = Configuration(timer_placement=TimerPlacement.AFTER, log_ids=DEFAULT_CSV_IDS)
    delays = compute_naive_extraneous_activity_delays(event_log, config)
    # Assert there are delays if not using calendars
    assert len(delays) == 0

    # Discover extraneous delays (BEFORE) without availability calendars
    event_log = read_csv_log("./tests/assets/LoanApp_delays.csv", DEFAULT_CSV_IDS)
    config = Configuration(timer_placement=TimerPlacement.BEFORE, log_ids=DEFAULT_CSV_IDS)
    delays = compute_naive_extraneous_activity_delays(event_log, config)
    # Assert there are delays if not using calendars
    assert 'Reject application' in delays
    assert delays['Reject application'].name == "fix"
    assert delays['Reject application'].mean == 600
    assert 'Design loan offer' in delays
    assert delays['Design loan offer'].name == "fix"
    assert delays['Design loan offer'].mean == 600
    assert 'Approve Loan Offer' in delays
    assert delays['Approve Loan Offer'].name == "fix"
    assert delays['Approve Loan Offer'].mean == 1200
    assert len(delays) == 3

    # Discover extraneous delays (AFTER) without availability calendars
    event_log = read_csv_log("./tests/assets/LoanApp_delays.csv", DEFAULT_CSV_IDS)
    config = Configuration(timer_placement=TimerPlacement.AFTER, log_ids=DEFAULT_CSV_IDS)
    delays = compute_naive_extraneous_activity_delays(event_log, config)
    # Assert there are delays if not using calendars
    assert 'Assess loan risk' in delays
    assert delays['Assess loan risk'].name == "fix"
    assert delays['Assess loan risk'].mean == 600
    assert 'Design loan offer' in delays
    assert delays['Design loan offer'].name == "fix"
    assert delays['Design loan offer'].mean == 1200
    assert len(delays) == 2


def test_compute_complex_extraneous_activity_delays_LoanApp():
    # Discover extraneous delays (BEFORE) without availability calendars
    event_log = read_csv_log("./tests/assets/LoanApp_no_delays.csv", DEFAULT_CSV_IDS)
    config = Configuration(timer_placement=TimerPlacement.BEFORE, log_ids=DEFAULT_CSV_IDS)
    delays = compute_complex_extraneous_activity_delays(event_log, config)
    # Assert there are delays if not using calendars
    assert len(delays) == 0

    # Discover extraneous delays (AFTER) without availability calendars
    event_log = read_csv_log("./tests/assets/LoanApp_no_delays.csv", DEFAULT_CSV_IDS)
    config = Configuration(timer_placement=TimerPlacement.AFTER, log_ids=DEFAULT_CSV_IDS)
    delays = compute_complex_extraneous_activity_delays(event_log, config)
    # Assert there are delays if not using calendars
    assert len(delays) == 0

    # Discover extraneous delays (BEFORE) without availability calendars
    event_log = read_csv_log("./tests/assets/LoanApp_delays.csv", DEFAULT_CSV_IDS)
    config = Configuration(timer_placement=TimerPlacement.BEFORE, log_ids=DEFAULT_CSV_IDS)
    delays = compute_complex_extraneous_activity_delays(event_log, config)
    # Assert there are delays if not using calendars
    assert 'Reject application' in delays
    assert delays['Reject application'].name == "fix"
    assert delays['Reject application'].mean == 600
    assert 'Design loan offer' in delays
    assert delays['Design loan offer'].name == "fix"
    assert delays['Design loan offer'].mean == 600
    assert 'Approve Loan Offer' in delays
    assert delays['Approve Loan Offer'].name == "fix"
    assert delays['Approve Loan Offer'].mean == 1200
    assert len(delays) == 3

    # Discover extraneous delays (AFTER) without availability calendars
    event_log = read_csv_log("./tests/assets/LoanApp_delays.csv", DEFAULT_CSV_IDS)
    config = Configuration(timer_placement=TimerPlacement.AFTER, log_ids=DEFAULT_CSV_IDS)
    delays = compute_complex_extraneous_activity_delays(event_log, config)
    # Assert there are delays if not using calendars
    assert 'Assess loan risk' in delays
    assert delays['Assess loan risk'].name == "fix"
    assert delays['Assess loan risk'].mean == 600
    assert 'Design loan offer' in delays
    assert delays['Design loan offer'].name == "fix"
    assert delays['Design loan offer'].mean == 1200
    assert len(delays) == 2


def test_compute_extraneous_activity_delays_naive_vs_complex():
    # Discover naive extraneous delays (BEFORE)
    event_log = read_csv_log("./tests/assets/event_log_2.csv", DEFAULT_CSV_IDS)
    config = Configuration(timer_placement=TimerPlacement.BEFORE, log_ids=DEFAULT_CSV_IDS)
    delays = compute_naive_extraneous_activity_delays(event_log, config)
    # Assert there are no delays if naive technique is used
    assert len(delays) == 0

    # Discover complex extraneous delays (BEFORE)
    event_log = read_csv_log("./tests/assets/event_log_2.csv", DEFAULT_CSV_IDS)
    config = Configuration(timer_placement=TimerPlacement.BEFORE, log_ids=DEFAULT_CSV_IDS)
    delays = compute_complex_extraneous_activity_delays(event_log, config)
    # Assert there are delays if complex technique is used
    assert 'D' in delays
    assert len(delays) == 1


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
