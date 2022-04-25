import subprocess

from extraneous_activity_delays.config import Configuration


def simulate_bpmn_model(model_path: str, output_path: str, config: Configuration):
    args = ['java', '-jar', config.PATH_BIMP,
            model_path,
            '-csv',
            output_path]
    completed_process = subprocess.run(args, check=True, stdout=subprocess.PIPE)
    message = f'\nShell debug information:' \
              f'\n\targs = {completed_process.args}' \
              f'\n\tstdout = {completed_process.stdout.__str__()}' \
              f'\n\tstderr = {completed_process.stderr.__str__()}'
    print(message)
