import subprocess

import pandas as pd
from lxml.etree import ElementTree

from extraneous_activity_delays.bpmn_enhancer import add_timers_to_bpmn_model
from extraneous_activity_delays.config import Configuration
from extraneous_activity_delays.delay_discoverer import calculate_extraneous_activity_delays


def enhance_bpmn_model_with_delays(event_log: pd.DataFrame, bpmn_document: ElementTree, configuration: Configuration):
    # Calculate base extraneous delays
    timers = calculate_extraneous_activity_delays(event_log, configuration.log_ids)
    # Enhance process model
    add_timers_to_bpmn_model(bpmn_document, timers)
    # Serialize to temporal BPMN file
    bpmn_document.write("temporal/model_path.bpmn", pretty_print=True)
    # Simulate with model
    _simulate_enhanced_model("temporal/model_path.bpmn", "temporal/log_path.csv")
    # Measure log distance


def _simulate_enhanced_model(model_path: str, output_path: str):
    args = ['java', '-jar', "bimp_executable",
            model_path,
            '-csv',
            output_path]
    completed_process = subprocess.run(args, check=True, stdout=subprocess.PIPE)
    message = f'\nShell debug information:' \
              f'\n\targs = {completed_process.args}' \
              f'\n\tstdout = {completed_process.stdout.__str__()}' \
              f'\n\tstderr = {completed_process.stderr.__str__()}'
    print(message)
