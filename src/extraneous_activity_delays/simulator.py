import enum
import subprocess

from extraneous_activity_delays.config import Configuration


class SimulationOutput(enum.Enum):
    SUCCESS = 1
    ERROR = 2


def simulate_bpmn_model(model_path: str, output_path: str, config: Configuration) -> SimulationOutput:
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
