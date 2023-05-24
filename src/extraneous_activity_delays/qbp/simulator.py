import subprocess

from start_time_estimator.config import EventLogIDs

from extraneous_activity_delays.config import Configuration, SimulationOutput

LOG_IDS = EventLogIDs(
    case="caseid",
    activity="task",
    start_time="start_timestamp",
    end_time="end_timestamp",
    resource="resource",
)


def simulate(model_path: str, output_path: str, config: Configuration) -> SimulationOutput:
    """
    Simulate the BPS model in [model_path] using QBP, writing the simulated log to [output_path].

    :param model_path:  path to the BPMN file with the simulation model.
    :param output_path: path to write the simulated log.
    :param config:      configuration settings.

    :return: the state of the simulation, either error or success.
    """
    args = ["java", "-jar", config.PATH_QBP, model_path, "-csv", output_path]
    # Run simulator
    completed_process = subprocess.run(args, check=True, stdout=subprocess.PIPE)
    stdout = completed_process.stdout.__str__()
    stderr = completed_process.stderr.__str__()
    # If debug on print outputs
    if config.debug:
        message = (
            f"\nShell debug information:"
            f"\n\targs = {completed_process.args}"
            f"\n\tstdout = {stdout}"
            f"\n\tstderr = {stderr}"
        )
        print(message)
    # Return standard output
    if "BPSimulatorException" in stdout and "Maximum allowed cycle time exceeded" in stdout:
        print("Warning! Simulation error due to max cycle time exceeded.")
        return SimulationOutput.ERROR
    else:
        return SimulationOutput.SUCCESS
