import copy
import json
from pathlib import Path
from statistics import mean
from typing import Union

import pandas as pd
from hyperopt import fmin, hp, tpe, STATUS_OK
from hyperopt.fmin import generate_trials_to_calculate

from extraneous_activity_delays.config import Configuration, SimulationOutput, SimulationModel, SimulationEngine
from extraneous_activity_delays.delay_discoverer import compute_extraneous_activity_delays
from extraneous_activity_delays.prosimos.infer_distribution import scale_distribution as scale_distribution_prosimos
from extraneous_activity_delays.prosimos.simulation_model_enhancer import \
    add_timers_to_simulation_model as add_timers_to_simulation_model_prosimos
from extraneous_activity_delays.prosimos.simulator import LOG_IDS as PROSIMOS_LOG_IDS
from extraneous_activity_delays.prosimos.simulator import simulate as simulate_prosimos
from extraneous_activity_delays.qbp.infer_distribution import scale_distribution as scale_distribution_qbp
from extraneous_activity_delays.qbp.simulation_model_enhancer import add_timers_to_simulation_model as add_timers_to_simulation_model_qbp
from extraneous_activity_delays.qbp.simulation_model_enhancer import set_number_instances_to_simulate, \
    set_start_datetime_to_simulate
from extraneous_activity_delays.qbp.simulator import LOG_IDS as QBP_LOG_IDS
from extraneous_activity_delays.qbp.simulator import simulate as simulate_qbp
from extraneous_activity_delays.utils import delete_folder, create_new_tmp_folder, split_log_training_validation_event_wise
from log_similarity_metrics.cycle_times import cycle_time_emd


class NaiveEnhancer:
    def __init__(self, event_log: pd.DataFrame, simulation_model: SimulationModel, configuration: Configuration):
        # Save parameters
        self.event_log = event_log
        self.simulation_model = simulation_model
        self.configuration = configuration
        self.log_ids = configuration.log_ids
        # Calculate extraneous delay timers
        self.timers = compute_extraneous_activity_delays(self.event_log, self.configuration, self.configuration.should_consider_timer)

    def enhance_simulation_model_with_delays(self) -> SimulationModel:
        # Enhance process model
        if self.configuration.simulation_engine == SimulationEngine.PROSIMOS:
            enhanced_simulation_model = add_timers_to_simulation_model_prosimos(self.simulation_model, self.timers)
        else:
            enhanced_simulation_model = add_timers_to_simulation_model_qbp(self.simulation_model, self.timers)
        # Return enhanced document
        return enhanced_simulation_model


class HyperOptEnhancer:
    def __init__(self, event_log: pd.DataFrame, simulation_model: SimulationModel, configuration: Configuration):
        # Save parameters
        self.event_log = event_log
        if configuration.training_partition_ratio is not None:
            # Train and validate the enhancement with hold-out
            self.training_log, self.validation_log = split_log_training_validation_event_wise(
                self.event_log,
                configuration.log_ids,
                configuration.training_partition_ratio
            )
        else:
            # Train and validate with the full event log
            self.training_log, self.validation_log = self.event_log.copy(), self.event_log.copy()
        self.simulation_model = copy.deepcopy(simulation_model)
        self.configuration = configuration
        self.log_ids = configuration.log_ids
        # Calculate extraneous delay timers
        self.timers = compute_extraneous_activity_delays(self.training_log, self.configuration, self.configuration.should_consider_timer)
        # Hyper-optimization search space
        if self.configuration.multi_parametrization:
            self.opt_space = {activity: hp.uniform(activity, 0.0, self.configuration.max_alpha) for activity in self.timers.keys()}
            baseline_iteration_params = [{activity: 1.0 for activity in self.timers.keys()}]
        else:
            self.opt_space = hp.uniform('alpha', 0.0, self.configuration.max_alpha)
            baseline_iteration_params = [{'alpha': 1.0}]
        # Variable to store the information of each optimization trial
        self.opt_trials = generate_trials_to_calculate(baseline_iteration_params)  # Force the first trial to be with this values
        # Result attributes
        self.best_timers = {}
        self.losses = []

    def enhance_simulation_model_with_delays(self) -> SimulationModel:
        if len(self.timers) > 0:
            # Launch hyper-optimization with the timers
            best_result = fmin(
                fn=self._enhancement_iteration,
                space=self.opt_space,
                algo=tpe.suggest,
                max_evals=self.configuration.num_iterations,
                trials=self.opt_trials,
                show_progressbar=False
            )
            # Remove all folders except best trial one
            for result in self.opt_trials.results:
                if result['output_folder'] != self.opt_trials.best_trial['result']['output_folder']:
                    delete_folder(result['output_folder'])
            # Process best parameters result
            if self.configuration.multi_parametrization:
                # One scale factor per timer
                best_alphas = {activity: round(best_result[activity], 2) for activity in best_result}
            else:
                # One scale factor for each timer
                best_alphas = {activity: round(best_result['alpha'], 2) for activity in self.timers.keys()}
            # Transform timers based on [best_alphas]
            scaled_timers = self._get_scaled_timers(best_alphas)
            self.best_timers = scaled_timers
            # Enhance process model
            if self.configuration.simulation_engine == SimulationEngine.PROSIMOS:
                enhanced_simulation_model = add_timers_to_simulation_model_prosimos(self.simulation_model, scaled_timers)
            else:
                enhanced_simulation_model = add_timers_to_simulation_model_qbp(self.simulation_model, scaled_timers)
        else:
            # No timers discovered, make a copy of current simulation model
            enhanced_simulation_model = self.simulation_model
        # Return enhanced document
        return enhanced_simulation_model

    def _enhancement_iteration(self, params: Union[float, dict]) -> dict:
        # Process params
        if self.configuration.multi_parametrization:
            # Dictionary with a scale factor for each activity, so let it be
            alphas = {activity: round(params[activity], 2) for activity in params}
        else:
            # Only one scale factor, create dictionary with that factor for each activity
            alphas = {activity: round(params, 2) for activity in self.timers.keys()}
        # Get iteration folder
        output_folder = create_new_tmp_folder(self.configuration.PATH_OUTPUTS)
        # Transform timers based on [alpha]
        scaled_timers = self._get_scaled_timers(alphas)
        # Enhance process model
        if self.configuration.simulation_engine == SimulationEngine.PROSIMOS:
            enhanced_simulation_model = add_timers_to_simulation_model_prosimos(self.simulation_model, scaled_timers)
        else:
            enhanced_simulation_model = add_timers_to_simulation_model_qbp(self.simulation_model, scaled_timers)
        # Evaluate candidate
        distance_value = self._evaluate_iteration(enhanced_simulation_model, output_folder, alphas, scaled_timers)
        self.losses += [distance_value]
        # Return response
        return {'loss': distance_value, 'status': STATUS_OK, 'output_folder': str(output_folder)}

    def _evaluate_iteration(self, simulation_model: SimulationModel, output_folder: Path, params: dict, iteration_timers: dict) -> float:
        # Create paths to serialize current simulation model
        bpmn_model_path = str(output_folder.joinpath("{}_enhanced.bpmn".format(self.configuration.process_name)))
        simulation_parameters_path = str(output_folder.joinpath("{}_enhanced.json".format(self.configuration.process_name)))
        num_cases_to_simulate = len(self.validation_log[self.log_ids.case].unique())
        simulation_starting_time = min(self.validation_log[self.log_ids.start_time])
        # Serialize
        if self.configuration.simulation_engine == SimulationEngine.PROSIMOS:
            # If simulation with Prosimos, Serialize BPMN model and parameters
            simulation_model.bpmn_document.write(bpmn_model_path, pretty_print=True)
            with open(simulation_parameters_path, 'w') as output_file:
                json.dump(simulation_model.simulation_parameters, output_file)
        else:
            # If simulation with QBP, set num instances and start time in the BPMN model, and serialize it
            set_number_instances_to_simulate(simulation_model.bpmn_document, num_cases_to_simulate)
            set_start_datetime_to_simulate(simulation_model.bpmn_document, simulation_starting_time)
            simulation_model.bpmn_document.write(bpmn_model_path, pretty_print=True)
        # Initialize list to store EMDs
        cycle_time_emds, metrics_report = [], []
        bin_size = max(  # Bin size for the cycle time EMD
            [events[self.log_ids.end_time].max() - events[self.log_ids.start_time].min()
             for case, events in self.validation_log.groupby(self.log_ids.case)]
        ) / 1000
        # IDs of the simulated logs depending on the engine
        sim_log_ids = PROSIMOS_LOG_IDS if self.configuration.simulation_engine == SimulationEngine.PROSIMOS else QBP_LOG_IDS
        # Simulate and measure quality
        for i in range(self.configuration.num_evaluation_simulations):
            # Simulate with model
            tmp_simulated_log_path = str(output_folder.joinpath("{}_simulated_{}.csv".format(self.configuration.process_name, i)))
            if self.configuration.simulation_engine == SimulationEngine.PROSIMOS:
                sim_out = simulate_prosimos(
                    model_path=bpmn_model_path,
                    parameters_path=simulation_parameters_path,
                    num_cases=num_cases_to_simulate,
                    starting_timestamp=simulation_starting_time,
                    output_path=tmp_simulated_log_path
                )
            else:
                sim_out = simulate_qbp(bpmn_model_path, tmp_simulated_log_path, self.configuration)
            # If the simulation didn't fail
            if sim_out == SimulationOutput.SUCCESS:
                # Read simulated event log
                simulated_event_log = pd.read_csv(tmp_simulated_log_path)
                simulated_event_log[sim_log_ids.start_time] = pd.to_datetime(simulated_event_log[sim_log_ids.start_time], utc=True)
                simulated_event_log[sim_log_ids.end_time] = pd.to_datetime(simulated_event_log[sim_log_ids.end_time], utc=True)
                # Measure log distance
                cycle_time_emd_value = cycle_time_emd(self.validation_log, self.log_ids, simulated_event_log, sim_log_ids, bin_size)
                cycle_time_emds += [cycle_time_emd_value]
                metrics_report += ["\tCycle time EMD {}: {}\n".format(i, cycle_time_emd_value)]
        # Get average
        if len(cycle_time_emds) > 0:
            mean_cycle_time_emd = mean(cycle_time_emds)
        else:
            # All simulations ended in error, set default value
            mean_cycle_time_emd = 1000000  # TODO replace by MAX_FLOAT?
        # Write metrics to file
        with open(output_folder.joinpath("metrics.txt"), 'a') as file:
            file.write("Iteration params: {}\n".format(params))
            file.write("\nTimers:\n")
            for activity in iteration_timers:
                file.write("\t'{}': {}\n".format(activity, iteration_timers[activity]))
            file.write("\nMetrics:\n")
            for line in metrics_report:
                file.write(line)
            file.write("\nMean cycle time EMD: {}\n".format(mean_cycle_time_emd))
        # Return metric
        return mean_cycle_time_emd

    def _get_scaled_timers(self, alphas: dict):
        scaled_timers = {}
        # For each timer
        for activity in self.timers:
            # If the scaling factor is not 0.0 create a timer
            if (activity in alphas) and (alphas[activity] > 0.0):
                if self.configuration.simulation_engine == SimulationEngine.PROSIMOS:
                    # Scale Prosimos timer
                    scaled_timers[activity] = scale_distribution_prosimos(self.timers[activity], alphas[activity])
                else:
                    # Scale QBP timer
                    scaled_timers[activity] = scale_distribution_qbp(self.timers[activity], alphas[activity])
        # Return timers
        return scaled_timers
