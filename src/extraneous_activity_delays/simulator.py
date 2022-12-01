import enum
import os
import subprocess

from external_tools.Prosimos.bpdfr_simulation_engine.simulation_engine import run_simulation
from extraneous_activity_delays.config import Configuration


class SimulationOutput(enum.Enum):
    SUCCESS = 1
    ERROR = 2


def simulate_bpmn_model_prosimos(
        model_path: str,
        parameters_path: str,
        num_cases: int,
        starting_timestamp: str,
        output_path: str,
        config: Configuration
) -> SimulationOutput:
    """
    Simulate the BPS model in [model_path] using BIMP, writing the simulated log to [output_path].

    :param model_path:          path to the BPMN file with the simulation model.
    :param parameters_path:     path to the JSON file with the parameters of the PROSIMOS simulation.
    :param num_cases:           number of cases to simulate.
    :param starting_timestamp:  timestamp with the point in time to start the simulation.
    :param output_path:         path to write the simulated log.
    :param config:              configuration settings.

    :return: the state of the simulation, either error or success.
    """
    # Run simulation
    run_simulation(
        bpmn_path=model_path,
        json_path=parameters_path,
        total_cases=num_cases,
        stat_out_path=None,  # No statistics
        log_out_path=output_path,
        starting_at=starting_timestamp,
        is_event_added_to_log=False  # Don't add Events (start/end/timers) to output log
    )
    # Return status of the simulation
    if os.path.isfile(output_path):
        return SimulationOutput.SUCCESS
    else:
        print("Warning! An error has occurred with the simulation, the output log has not been created.")
        return SimulationOutput.ERROR


def simulate_bpmn_model_bimp(model_path: str, output_path: str, config: Configuration) -> SimulationOutput:
    """
    Simulate the BPS model in [model_path] using BIMP, writing the simulated log to [output_path].

    :param model_path:  path to the BPMN file with the simulation model.
    :param output_path: path to write the simulated log.
    :param config:      configuration settings.

    :return: the state of the simulation, either error or success.
    """
    args = ['java', '-jar', config.PATH_BIMP,
            model_path,
            '-csv',
            output_path]
    # Run simulator
    completed_process = subprocess.run(args, check=True, stdout=subprocess.PIPE)
    stdout = completed_process.stdout.__str__()
    stderr = completed_process.stderr.__str__()
    # If debug on print outputs
    if config.debug:
        message = f'\nShell debug information:' \
                  f'\n\targs = {completed_process.args}' \
                  f'\n\tstdout = {stdout}' \
                  f'\n\tstderr = {stderr}'
        print(message)
    # Return standard output
    if "BPSimulatorException" in stdout and "Maximum allowed cycle time exceeded" in stdout:
        print("Warning! Simulation error due to max cycle time exceeded.")
        return SimulationOutput.ERROR
    else:
        return SimulationOutput.SUCCESS
