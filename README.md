# Extraneous activity delays BPS model enhancer

Python implementation of the approach to enhance BPS models by modeling the extraneous activity delays with timer events.
This technique has been presented in the paper "Modeling Extraneous Activity Delays in Business Process Simulation", by David Chapela-Campa and Marlon Dumas.


The technique takes as input a BPS model (in BPMN format) and an event log (pd.DataFrame) recording the execution of the activities of a process (including resource information).
The approach discovers from the event log the waiting time previous to each activity caused by extraneous factors and adds a timer event previous to each activity modeling this delay.
The result is an enhanced version of the input BPS model with timer events to model the extraneous waiting time.

## Requirements

- **Python v3.9.5+**
- **PIP v21.1.2+**
- Python dependencies: The packages listed in `requirements.txt`.

## Basic Usage

Check the [synthetic](https://github.com/AutomatedProcessImprovement/extraneous-activity-delays/blob/main/tests/synthetic-evaluation.py) and [real-life](https://github.com/AutomatedProcessImprovement/extraneous-activity-delays/blob/main/tests/real-life-evaluation.py) evaluation files for an example of the different executions of the technique, and [config file](https://github.com/AutomatedProcessImprovement/extraneous-activity-delays/blob/main/src/extraneous_activity_delays/config.py) for an explanation of the configuration parameters.
However, here we provide a simple example of the hyper-parameter optimization version of the proposal.

```python
# Set up default configuration
log_ids = DEFAULT_CSV_IDS
config = Configuration(
        log_ids=log_ids, process_name="example",
        max_alpha=50.0, training_partition_ratio=0.5, num_evaluations=200
    )
# Read event log
event_log = pd.read_csv("path_to_input_log.csv")
event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time], utc=True)
event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True)
event_log[log_ids.resource].fillna("NOT_SET", inplace=True)
event_log[log_ids.resource] = event_log[log_ids.resource].astype("string")
# Read BPMN model
parser = etree.XMLParser(remove_blank_text=True)
bpmn_model = etree.parse("path_to_bps_model.bpmn", parser)
set_number_instances_to_simulate(bpmn_model, len(event_log[config.log_ids.case].unique()))
set_start_datetime_to_simulate(bpmn_model, min(event_log[config.log_ids.start_time]))
# Enhance with hyper-parametrized activity delays with hold-out
enhancer = HyperOptEnhancer(event_log, bpmn_model, config)
enhanced_bpmn_model = enhancer.enhance_bpmn_model_with_delays()
# Write enhanced BPS model
enhanced_bpmn_model.write("path_of_enhanced_bps_model.bpmn", pretty_print=True)
```
