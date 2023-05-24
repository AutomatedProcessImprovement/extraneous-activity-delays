import datetime
import os

from prosimos.simulation_engine import run_simulation
from start_time_estimator.config import EventLogIDs

from extraneous_activity_delays.config import SimulationOutput

LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    start_time="start_time",
    end_time="end_time",
    resource="resource",
)


def simulate(
    model_path: str,
    parameters_path: str,
    num_cases: int,
    starting_timestamp: datetime.datetime,
    output_path: str,
    record_events: bool = False,
) -> SimulationOutput:
    """
    Simulate the BPS model in [model_path] with the simulation parameters in [parameters_path] using PROSIMOS. The simulated log
    os written to [output_path].

    :param model_path:          path to the BPMN file with the simulation model.
    :param parameters_path:     path to the JSON file with the parameters of the PROSIMOS simulation.
    :param num_cases:           number of cases to simulate.
    :param starting_timestamp:  timestamp with the point in time to start the simulation.
    :param output_path:         path to write the simulated log.
    :param record_events:       if True, the simulated log will include the events (e.g., timer events).

    :return: the state of the simulation, either error or success.
    """
    # Run simulation
    run_simulation(
        bpmn_path=model_path,
        json_path=parameters_path,
        total_cases=num_cases,
        stat_out_path=None,  # No statistics
        log_out_path=output_path,
        starting_at=(
            starting_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")
            + starting_timestamp.strftime("%z")[:-2]
            + ":"
            + starting_timestamp.strftime("%z")[-2:]
        ),
        is_event_added_to_log=record_events,  # Don't add Events (start/end/timers) to output log
    )
    # Return status of the simulation
    if os.path.isfile(output_path):
        return SimulationOutput.SUCCESS
    else:
        print("Warning! An error has occurred during PROSIMOS simulation: the output log has not been created.")
        return SimulationOutput.ERROR
