import json

from lxml import etree

from estimate_start_times.config import DEFAULT_CSV_IDS
from extraneous_activity_delays.config import Configuration, SimulationEngine, SimulationModel, OptimizationMetric
from extraneous_activity_delays.enhance_with_delays import HyperOptEnhancer
from pix_utils.input import read_csv_log


def optimize_with_prosimos():
    # Set up configuration with PROSIMOS
    config = Configuration(
        log_ids=DEFAULT_CSV_IDS, process_name="prosimos-example",
        max_alpha=5.0,
        num_iterations=10, simulation_engine=SimulationEngine.PROSIMOS,
        optimization_metric=OptimizationMetric.RELATIVE_EMD
    )
    # Read event log
    event_log = read_csv_log("./assets/prosimos-loan-app/LoanApp_sequential_9-5_diffres.csv.gz", config.log_ids)
    # Read BPMN model
    parser = etree.XMLParser(remove_blank_text=True)
    bpmn_model = etree.parse("./assets/prosimos-loan-app/LoanApp_sequential_9-5_diffres.bpmn", parser)
    # Read simulation parameters
    with open("./assets/prosimos-loan-app/LoanApp_sequential_9-5_diffres.json") as json_file:
        simulation_parameters = json.load(json_file)
    simulation_model = SimulationModel(bpmn_model, simulation_parameters)
    # Enhance with hyper-parametrized activity delays with hold-out
    enhancer = HyperOptEnhancer(event_log, simulation_model, config)
    enhanced_simulation_model = enhancer.enhance_simulation_model_with_delays()
    # Write enhanced BPS model
    enhanced_simulation_model.bpmn_document.write("../outputs/LoanApp_sequential_9-5_diffres.bpmn", pretty_print=True)


if __name__ == '__main__':
    optimize_with_prosimos()
