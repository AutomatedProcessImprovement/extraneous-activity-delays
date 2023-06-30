import copy
import datetime
import json
from pathlib import Path
from statistics import mean
from typing import Union

import pandas as pd
from hyperopt import fmin, hp, tpe, STATUS_OK
from hyperopt.fmin import generate_trials_to_calculate
from log_distance_measures.absolute_event_distribution import (
    absolute_event_distribution_distance,
)
from log_distance_measures.circadian_event_distribution import (
    circadian_event_distribution_distance,
)
from log_distance_measures.cycle_time_distribution import (
    cycle_time_distribution_distance,
)
from log_distance_measures.relative_event_distribution import (
    relative_event_distribution_distance,
)
from pix_framework.log_split.log_split import split_log_training_validation_event_wise
from start_time_estimator.config import EventLogIDs

from extraneous_activity_delays.config import (
    Configuration,
    SimulationOutput,
    SimulationModel,
    SimulationEngine,
    OptimizationMetric,
    DiscoveryMethod,
)
from extraneous_activity_delays.delay_discoverer import (
    compute_naive_extraneous_activity_delays,
    compute_complex_extraneous_activity_delays,
)
from extraneous_activity_delays.prosimos.simulation_model_enhancer import (
    add_timers_to_simulation_model as add_timers_to_simulation_model_prosimos,
)
from extraneous_activity_delays.prosimos.simulator import LOG_IDS as PROSIMOS_LOG_IDS
from extraneous_activity_delays.prosimos.simulator import simulate as simulate_prosimos
from extraneous_activity_delays.qbp.simulation_model_enhancer import (
    add_timers_to_simulation_model as add_timers_to_simulation_model_qbp,
)
from extraneous_activity_delays.qbp.simulation_model_enhancer import (
    set_number_instances_to_simulate,
    set_start_datetime_to_simulate,
)
from extraneous_activity_delays.qbp.simulator import LOG_IDS as QBP_LOG_IDS
from extraneous_activity_delays.qbp.simulator import simulate as simulate_qbp
from extraneous_activity_delays.utils.file_manager import (
    delete_folder,
    create_new_tmp_folder,
)


class DirectEnhancer:
    def __init__(
        self,
        event_log: pd.DataFrame,
        simulation_model: SimulationModel,
        configuration: Configuration,
    ):
        # Save parameters
        self.event_log = event_log
        self.simulation_model = simulation_model
        self.configuration = configuration
        self.log_ids = configuration.log_ids
        # Compute extraneous delay timers
        if self.configuration.discovery_method == DiscoveryMethod.NAIVE:
            self.timers = compute_naive_extraneous_activity_delays(
                self.event_log,
                self.configuration,
                self.configuration.should_consider_timer,
            )
        elif self.configuration.discovery_method == DiscoveryMethod.COMPLEX:
            self.timers = compute_complex_extraneous_activity_delays(
                self.event_log,
                self.configuration,
                self.configuration.should_consider_timer,
            )
        else:
            raise ValueError("Invalid delay discovery method selected!")

    def enhance_simulation_model_with_delays(self) -> SimulationModel:
        # Enhance process model
        if self.configuration.simulation_engine == SimulationEngine.PROSIMOS:
            enhanced_simulation_model = add_timers_to_simulation_model_prosimos(
                self.simulation_model, self.timers, self.configuration.timer_placement
            )
        else:
            enhanced_simulation_model = add_timers_to_simulation_model_qbp(
                self.simulation_model, self.timers, self.configuration.timer_placement
            )
        # Return enhanced document
        return enhanced_simulation_model


class HyperOptEnhancer:
    def __init__(
        self,
        event_log: pd.DataFrame,
        simulation_model: SimulationModel,
        configuration: Configuration,
    ):
        # Save parameters
        self.event_log = event_log
        if configuration.training_partition_ratio is not None:
            # Train and validate the enhancement with hold-out
            (
                self.training_log,
                self.validation_log,
            ) = split_log_training_validation_event_wise(
                self.event_log,
                configuration.log_ids,
                configuration.training_partition_ratio,
            )
        else:
            # Train and validate with the full event log
            self.training_log, self.validation_log = (
                self.event_log.copy(),
                self.event_log.copy(),
            )
        self.simulation_model = copy.deepcopy(simulation_model)
        self.configuration = configuration
        self.log_ids = configuration.log_ids
        # Compute extraneous delay timers
        if self.configuration.discovery_method == DiscoveryMethod.NAIVE:
            self.timers = compute_naive_extraneous_activity_delays(
                self.event_log,
                self.configuration,
                self.configuration.should_consider_timer,
            )
        elif self.configuration.discovery_method == DiscoveryMethod.COMPLEX:
            self.timers = compute_complex_extraneous_activity_delays(
                self.event_log,
                self.configuration,
                self.configuration.should_consider_timer,
            )
        else:
            raise ValueError("Invalid delay discovery method selected!")
        # Hyper-optimization search space
        self.opt_space = {
            activity: hp.uniform(activity, 0.0, self.configuration.max_alpha) for activity in self.timers.keys()
        }
        baseline_iteration_params = [
            {activity: 0.0 for activity in self.timers.keys()},  # No timers
            {activity: 1.0 for activity in self.timers.keys()},  # Discovered timers
        ]
        # Variable to store the information of each optimization trial
        self.opt_trials = generate_trials_to_calculate(
            baseline_iteration_params
        )  # Force the first trial to be with this values
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
                show_progressbar=False,
            )
            # Remove all folders except best trial one
            if self.configuration.clean_intermediate_files:
                for result in self.opt_trials.results:
                    if result["output_folder"] != self.opt_trials.best_trial["result"]["output_folder"]:
                        delete_folder(result["output_folder"])
            # Process the best parameters result
            best_alphas = {activity: round(best_result[activity], 2) for activity in best_result}
            # Transform timers based on [best_alphas]
            scaled_timers = self._get_scaled_timers(best_alphas)
            self.best_timers = scaled_timers
            # Enhance process model
            if self.configuration.simulation_engine == SimulationEngine.PROSIMOS:
                enhanced_simulation_model = add_timers_to_simulation_model_prosimos(
                    self.simulation_model,
                    scaled_timers,
                    self.configuration.timer_placement,
                )
            else:
                enhanced_simulation_model = add_timers_to_simulation_model_qbp(
                    self.simulation_model,
                    scaled_timers,
                    self.configuration.timer_placement,
                )
        else:
            # No timers discovered, make a copy of current simulation model
            enhanced_simulation_model = self.simulation_model
        # Return enhanced document
        return enhanced_simulation_model

    def _enhancement_iteration(self, params: Union[float, dict]) -> dict:
        # Process params
        alphas = {activity: round(params[activity], 2) for activity in params}
        # Get iteration folder
        output_folder = create_new_tmp_folder(self.configuration.PATH_OUTPUTS)
        # Transform timers based on [alpha]
        scaled_timers = self._get_scaled_timers(alphas)
        # Enhance process model
        if self.configuration.simulation_engine == SimulationEngine.PROSIMOS:
            enhanced_simulation_model = add_timers_to_simulation_model_prosimos(
                self.simulation_model, scaled_timers, self.configuration.timer_placement
            )
        else:
            enhanced_simulation_model = add_timers_to_simulation_model_qbp(
                self.simulation_model, scaled_timers, self.configuration.timer_placement
            )
        # Evaluate candidate
        distance_value = self._evaluate_iteration(enhanced_simulation_model, output_folder, alphas, scaled_timers)
        self.losses += [distance_value]
        # Return response
        return {
            "loss": distance_value,
            "status": STATUS_OK,
            "output_folder": str(output_folder),
        }

    def _evaluate_iteration(
        self,
        simulation_model: SimulationModel,
        output_folder: Path,
        params: dict,
        iteration_timers: dict,
    ) -> float:
        # Create paths to serialize current simulation model
        bpmn_model_path = str(output_folder.joinpath("{}_enhanced.bpmn".format(self.configuration.process_name)))
        simulation_parameters_path = str(
            output_folder.joinpath("{}_enhanced.json".format(self.configuration.process_name))
        )
        num_cases_to_simulate = len(self.validation_log[self.log_ids.case].unique())
        simulation_starting_time = min(self.validation_log[self.log_ids.start_time])
        # Serialize
        if self.configuration.simulation_engine == SimulationEngine.PROSIMOS:
            # If simulation with Prosimos, Serialize BPMN model and parameters
            simulation_model.bpmn_document.write(bpmn_model_path, pretty_print=True)
            with open(simulation_parameters_path, "w") as output_file:
                json.dump(simulation_model.simulation_parameters, output_file)
        else:
            # If simulation with QBP, set num instances and start time in the BPMN model, and serialize it
            set_number_instances_to_simulate(simulation_model.bpmn_document, num_cases_to_simulate)
            set_start_datetime_to_simulate(simulation_model.bpmn_document, simulation_starting_time)
            simulation_model.bpmn_document.write(bpmn_model_path, pretty_print=True)
        # Initialize list to store EMDs
        performance_metrics, metrics_report = [], []
        bin_size = (
            max(  # Bin size for the performance metric (only cycle time EMD)
                [
                    events[self.log_ids.end_time].max() - events[self.log_ids.start_time].min()
                    for case, events in self.validation_log.groupby(self.log_ids.case)
                ]
            )
            / 1000
        )
        # IDs of the simulated logs depending on the engine
        sim_log_ids = (
            PROSIMOS_LOG_IDS if self.configuration.simulation_engine == SimulationEngine.PROSIMOS else QBP_LOG_IDS
        )
        # Simulate and measure quality
        for i in range(self.configuration.num_evaluation_simulations):
            # Simulate with model
            tmp_simulated_log_path = str(
                output_folder.joinpath("{}_simulated_{}.csv".format(self.configuration.process_name, i))
            )
            if self.configuration.simulation_engine == SimulationEngine.PROSIMOS:
                sim_out = simulate_prosimos(
                    model_path=bpmn_model_path,
                    parameters_path=simulation_parameters_path,
                    num_cases=num_cases_to_simulate,
                    starting_timestamp=simulation_starting_time,
                    output_path=tmp_simulated_log_path,
                )
            else:
                sim_out = simulate_qbp(bpmn_model_path, tmp_simulated_log_path, self.configuration)
            # If the simulation didn't fail
            if sim_out == SimulationOutput.SUCCESS:
                # Read simulated event log
                simulated_event_log = pd.read_csv(tmp_simulated_log_path)
                simulated_event_log[sim_log_ids.start_time] = pd.to_datetime(
                    simulated_event_log[sim_log_ids.start_time], utc=True
                )
                simulated_event_log[sim_log_ids.end_time] = pd.to_datetime(
                    simulated_event_log[sim_log_ids.end_time], utc=True
                )
                # Measure log distance
                performance_value = self._compute_performance_metric(simulated_event_log, sim_log_ids, bin_size)
                performance_metrics += [performance_value]
                metrics_report += ["\tPerformance metric {}: {}\n".format(i, performance_value)]
        # Get average
        if len(performance_metrics) > 0:
            mean_performance_metric = mean(performance_metrics)
        else:
            # All simulations ended in error, set default value
            mean_performance_metric = 1000000  # TODO replace by MAX_FLOAT?
        # Write metrics to file
        with open(output_folder.joinpath("metrics.txt"), "a") as file:
            file.write("Iteration params: {}\n".format(params))
            file.write("\nTimers:\n")
            for activity in iteration_timers:
                file.write("\t'{}': {}\n".format(activity, iteration_timers[activity]))
            file.write("\nMetrics:\n")
            for line in metrics_report:
                file.write(line)
            file.write("\nMean performance metric: {}\n".format(mean_performance_metric))
        # Return metric
        return mean_performance_metric

    def _compute_performance_metric(
        self,
        simulated_event_log: pd.DataFrame,
        sim_log_ids: EventLogIDs,
        bin_size: datetime.timedelta = datetime.timedelta(hours=1),
    ) -> float:
        if self.configuration.optimization_metric is OptimizationMetric.CYCLE_TIME:
            return cycle_time_distribution_distance(
                self.validation_log,
                self.log_ids,
                simulated_event_log,
                sim_log_ids,
                bin_size,
            )
        elif self.configuration.optimization_metric is OptimizationMetric.CIRCADIAN_EMD:
            return circadian_event_distribution_distance(
                self.validation_log, self.log_ids, simulated_event_log, sim_log_ids
            )
        elif self.configuration.optimization_metric is OptimizationMetric.ABSOLUTE_EMD:
            return absolute_event_distribution_distance(
                self.validation_log, self.log_ids, simulated_event_log, sim_log_ids
            )
        elif self.configuration.optimization_metric is OptimizationMetric.RELATIVE_EMD:
            return relative_event_distribution_distance(
                self.validation_log, self.log_ids, simulated_event_log, sim_log_ids
            )
        else:
            print("WARNING: Unknown optimization metric! Optimizing for cycle time!")
            return cycle_time_distribution_distance(
                self.validation_log,
                self.log_ids,
                simulated_event_log,
                sim_log_ids,
                bin_size,
            )

    def _get_scaled_timers(self, alphas: dict):
        scaled_timers = {}
        # For each timer
        for activity in self.timers:
            # If the scaling factor is not 0.0 create a timer
            if (activity in alphas) and (alphas[activity] > 0.0):
                scaled_timers[activity] = self.timers[activity].scale_distribution(alphas[activity])
        # Return timers
        return scaled_timers
