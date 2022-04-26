import datetime
import os
import shutil
import uuid
from pathlib import Path
from statistics import mean
from typing import Union

import pandas as pd
from estimate_start_times.config import EventLogIDs
from hyperopt import fmin, hp, Trials, tpe, STATUS_OK
from lxml.etree import ElementTree

from extraneous_activity_delays.bpmn_enhancer import add_timers_to_bpmn_model
from extraneous_activity_delays.config import Configuration, OptimizationSpaceType
from extraneous_activity_delays.delay_discoverer import calculate_extraneous_activity_delays
from extraneous_activity_delays.infer_distribution import scale_distribution
from extraneous_activity_delays.metrics import trace_duration_emd
from extraneous_activity_delays.simulator import simulate_bpmn_model


class Enhancer:
    def __init__(self, event_log: pd.DataFrame, bpmn_document: ElementTree, configuration: Configuration):
        # Save parameters
        self.event_log = event_log
        self.bpmn_document = bpmn_document
        self.configuration = configuration
        self.log_ids = configuration.log_ids
        # Calculate extraneous delay timers
        self.timers = calculate_extraneous_activity_delays(self.event_log, self.log_ids)
        # Variable to store the information of each optimization trial
        self.opt_trials = Trials()
        # Hyper-optimization search space: one scale factor (float from 0 to 1) per activity
        if self.configuration.optimization_space == OptimizationSpaceType.SINGLE_FACTOR:
            self.opt_space = hp.uniform('alpha', 0, 1)
        else:
            self.opt_space = {activity: hp.uniform(activity, 0, 1) for activity in self.timers.keys()}

    def enhance_bpmn_model_with_delays(self) -> ElementTree:
        # Launch hyper-optimization with the timers
        best_alphas = fmin(
            fn=self._enhancement_iteration,
            space=self.opt_space,
            algo=tpe.suggest,
            max_evals=self.configuration.num_evaluations,
            trials=self.opt_trials,
            show_progressbar=False
        )
        # Remove all folders except best trial one
        for result in self.opt_trials.results:
            if result['output_folder'] != self.opt_trials.best_trial['result']['output_folder']:
                shutil.rmtree(result['output_folder'], ignore_errors=True)  # TODO externalize to utils.py or something like that
        # If only one scale factor create dictionary with that factor for each activity
        if self.configuration.optimization_space == OptimizationSpaceType.SINGLE_FACTOR:
            best_alphas = {activity: best_alphas['alpha'] for activity in self.timers.keys()}
        # Transform timers based on [best_alphas]
        scaled_timers = {activity: scale_distribution(self.timers[activity], best_alphas[activity]) for activity in self.timers}
        # Enhance process model
        enhanced_document = add_timers_to_bpmn_model(self.bpmn_document, scaled_timers)
        # Return enhanced document
        return enhanced_document

    def _enhancement_iteration(self, alphas: Union[float, dict]) -> dict:
        # If only one scale factor create dictionary with that factor for each activity
        if self.configuration.optimization_space == OptimizationSpaceType.SINGLE_FACTOR:
            alphas = {activity: alphas for activity in self.timers.keys()}
        # Get iteration folder
        output_folder = self.configuration.PATH_OUTPUTS.joinpath(
            datetime.datetime.today().strftime('%Y%m%d_') + str(uuid.uuid4()).upper().replace('-', '_')
        )
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)  # TODO externalize to utils.py file or something like that
        # Transform timers based on [alpha]
        scaled_timers = {activity: scale_distribution(self.timers[activity], alphas[activity]) for activity in self.timers}
        # Enhance process model
        enhanced_bpmn_document = add_timers_to_bpmn_model(self.bpmn_document, scaled_timers)
        # Serialize to temporal BPMN file
        tmp_model_path = str(output_folder.joinpath("enhanced_model.bpmn"))
        enhanced_bpmn_document.write(tmp_model_path, pretty_print=True)
        # Evaluate candidate
        cycle_time_emd = self._evaluate(tmp_model_path, output_folder)
        # TODO write results (EMDs) to a file in output folder
        # Return response
        return {'loss': cycle_time_emd, 'status': STATUS_OK, 'output_folder': str(output_folder)}

    def _evaluate(self, bpmn_model_path: str, output_folder: Path) -> float:
        # EMDs of the simulations
        cycle_time_emds = []
        # IDs of the simulated logs from BIMP
        simulated_log_ids = EventLogIDs(
            case="caseid",
            activity="task",
            start_time="start_timestamp",
            end_time="end_timestamp",
            resource="resource"
        )
        # Bin size for the cycle time EMD
        bin_size = max(
            [events[self.log_ids.end_time].max() - events[self.log_ids.start_time].min()
             for case, events in self.event_log.groupby([self.log_ids.case])]
        ) / 1000
        # Simulate and measure quality
        for i in range(self.configuration.num_evaluation_simulations):
            # Simulate with model
            tmp_simulated_log_path = str(output_folder.joinpath("simulated_log_{}.csv".format(i)))
            simulate_bpmn_model(bpmn_model_path, tmp_simulated_log_path, self.configuration)
            # Read simulated event log
            simulated_event_log = pd.read_csv(tmp_simulated_log_path)
            simulated_event_log[simulated_log_ids.start_time] = pd.to_datetime(simulated_event_log[simulated_log_ids.start_time], utc=True)
            simulated_event_log[simulated_log_ids.end_time] = pd.to_datetime(simulated_event_log[simulated_log_ids.end_time], utc=True)
            # Measure log distance
            cycle_time_emds += [trace_duration_emd(self.event_log, self.log_ids, simulated_event_log, simulated_log_ids, bin_size)]
        # Return metric
        return mean(cycle_time_emds)
