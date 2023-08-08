import json

from extraneous_activity_delays.config import (
    Configuration,
    DiscoveryMethod,
    OptimizationMetric,
    SimulationEngine,
    SimulationModel,
    TimerPlacement,
)
from extraneous_activity_delays.enhance_with_delays import HyperOptEnhancer
from extraneous_activity_delays.prosimos.simulator import simulate
from lxml import etree
from pix_framework.io.event_log import DEFAULT_CSV_IDS, read_csv_log


def optimize_with_prosimos():
    # Set up configuration with PROSIMOS
    config = Configuration(
        log_ids=DEFAULT_CSV_IDS,
        process_name="prosimos-example",
        max_alpha=5.0,
        num_iterations=10,
        simulation_engine=SimulationEngine.PROSIMOS,
        optimization_metric=OptimizationMetric.RELATIVE_EMD,
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


def enhance_and_test():
    # Read event logs
    train_log = read_csv_log("../inputs/real-life/BPIC_2017_W_contained_Jun20_Sep16.csv.gz", DEFAULT_CSV_IDS)
    test_log = read_csv_log("../inputs/real-life/BPIC_2017_W_contained_Sep17_Dec19.csv.gz", DEFAULT_CSV_IDS)
    # Read simulation model
    parser = etree.XMLParser(remove_blank_text=True)
    bpmn_model = etree.parse("../inputs/real-life/prosimos-models/BPI_Challenge_2017.bpmn", parser)
    with open("../inputs/real-life/prosimos-models/BPI_Challenge_2017.json") as json_file:
        simulation_parameters = json.load(json_file)
    simulation_model = SimulationModel(bpmn_model, simulation_parameters)
    # Configurations
    config = Configuration(
        log_ids=DEFAULT_CSV_IDS,
        process_name="BPIC17",
        max_alpha=10.0,
        num_iterations=100,
        num_evaluation_simulations=5,
        discovery_method=DiscoveryMethod.NAIVE,
        timer_placement=TimerPlacement.AFTER,
        simulation_engine=SimulationEngine.PROSIMOS,
        optimization_metric=OptimizationMetric.RELATIVE_EMD,
    )
    # Discover extraneous delays
    # enhancer = DirectEnhancer(train_log, simulation_model, config)
    enhancer = HyperOptEnhancer(train_log, simulation_model, config)
    enhanced_simulation_model = enhancer.enhance_simulation_model_with_delays()
    enhanced_simulation_model.bpmn_document.write("../outputs/BPI_Challenge_2017_timers.bpmn", pretty_print=True)
    with open("../outputs/BPI_Challenge_2017_timers.json", "w") as f:
        json.dump(enhanced_simulation_model.simulation_parameters, f)
    for i in range(10):
        # Simulate against test
        simulated_log_path = "../outputs/BPI_Challenge_2017_sim_timers_{}.csv".format(i)
        simulate(
            model_path="../outputs/BPI_Challenge_2017_timers.bpmn",
            parameters_path="../outputs/BPI_Challenge_2017_timers.json",
            num_cases=test_log["case_id"].nunique(),
            starting_timestamp=test_log["start_time"].min(),
            output_path=simulated_log_path,
        )


if __name__ == "__main__":
    optimize_with_prosimos()
