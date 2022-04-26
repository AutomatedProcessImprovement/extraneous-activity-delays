import copy
import datetime
import os
import uuid

import pandas as pd
from estimate_start_times.config import EventLogIDs
from hyperopt import fmin, hp, Trials, tpe, STATUS_OK
from lxml.etree import ElementTree

from extraneous_activity_delays.bpmn_enhancer import add_timers_to_bpmn_model
from extraneous_activity_delays.config import Configuration
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
        # Variable to store the information of each optimization trial
        self.opt_trials = Trials()
        # Calculate extraneous delay timers
        self.timers = calculate_extraneous_activity_delays(self.event_log, self.log_ids)

    def enhance_bpmn_model_with_delays(self) -> ElementTree:
        # Launch hyper-optimization with the timers
        best = fmin(
            fn=self._enhancement_iteration,
            space=hp.uniform("alpha", 0, 1),
            algo=tpe.suggest,
            max_evals=10,
            trials=self.opt_trials,
            show_progressbar=False
        )
        # TODO remove all non best folders
        # Transform timers based on best [alpha]
        scaled_timers = {activity: scale_distribution(self.timers[activity], best['alpha']) for activity in self.timers}
        # Enhance process model
        enhanced_document = add_timers_to_bpmn_model(self.bpmn_document, scaled_timers)
        # Return enhanced document
        return enhanced_document

    def _enhancement_iteration(self, alpha: float) -> dict:
        # Get iteration folder
        output_folder = self.configuration.PATH_OUTPUTS.joinpath(
            datetime.datetime.today().strftime('%Y%m%d_') + str(uuid.uuid4()).upper().replace('-', '_')
        )
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)  # TODO externalize to utils.py file or something like that
        # Transform timers based on [alpha]
        # TODO improve with one [alpha] per activity
        scaled_timers = {activity: scale_distribution(self.timers[activity], alpha) for activity in self.timers}
        # Enhance process model
        enhanced_bpmn_document = add_timers_to_bpmn_model(self.bpmn_document, scaled_timers)
        # Serialize to temporal BPMN file
        tmp_model_path = str(output_folder.joinpath("enhanced_model.bpmn"))
        enhanced_bpmn_document.write(tmp_model_path, pretty_print=True)
        # Simulate with model
        # TODO simulate 5 times and get the mean
        tmp_simulated_log_path = str(output_folder.joinpath("simulated_log.csv"))
        simulate_bpmn_model(tmp_model_path, tmp_simulated_log_path, self.configuration)
        # Read simulated event log
        simulated_log_ids = EventLogIDs(
            case="caseid",
            activity="task",
            start_time="start_timestamp",
            end_time="end_timestamp",
            resource="resource"
        )
        simulated_event_log = pd.read_csv(tmp_simulated_log_path)
        simulated_event_log[simulated_log_ids.start_time] = pd.to_datetime(simulated_event_log[simulated_log_ids.start_time], utc=True)
        simulated_event_log[simulated_log_ids.end_time] = pd.to_datetime(simulated_event_log[simulated_log_ids.end_time], utc=True)
        # Measure log distance
        bin_size = max(
            [events[self.log_ids.end_time].max() - events[self.log_ids.start_time].min()
             for case, events in self.event_log.groupby([self.log_ids.case])]
        ) / 1000
        cycle_time_emd = trace_duration_emd(self.event_log, self.log_ids, simulated_event_log, simulated_log_ids, bin_size)
        # Return response
        return {'loss': cycle_time_emd, 'status': STATUS_OK}
