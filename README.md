# Extraneous activity delays BPS model enhancer

![build](https://github.com/AutomatedProcessImprovement/extraneous-activity-delays/actions/workflows/build.yaml/badge.svg)
![version](https://img.shields.io/github/v/tag/AutomatedProcessImprovement/extraneous-activity-delays)

Python implementation of the approach to enhance BPS models by modeling the extraneous activity delays with timer events. A preliminary
version of this technique has been presented in the paper **"Modeling Extraneous Activity Delays in Business Process Simulation"**, by David
Chapela-Campa and Marlon Dumas. The complete version is presented in the paper **"Enhancing Business Process Simulation Models with
Extraneous Activity Delays"**, by David Chapela-Campa and Marlon Dumas.

The technique supports two simulation engines: QBP and Prosimos. The input consists of *i)* an event log (pd.DataFrame) recording the
execution of the activities of a process (including resource information), and *ii)* a BPS model. In QBP, the BPS model is represented by a
BPMN file (example [here](https://github.com/AutomatedProcessImprovement/extraneous-activity-delays/tree/main/inputs/real-life/qbp-models)).
In Prosimos, by a BPMN file with the process model structure, and JSON file with the simulation parameters (example
[here](https://github.com/AutomatedProcessImprovement/extraneous-activity-delays/tree/main/inputs/real-life/prosimos-models)).

The approach uses the input event log to discover the waiting time caused by extraneous factors, and enhances the input BPS model with timer
events to model such delays. The proposal consists of different configurations such as:

- Discover the extraneous delays using the Naive proposal (`DiscoveryMethod.NAIVE`), or the Eclipse-aware proposal
  (`DiscoveryMethod.COMPLEX`).
- Consider the extraneous delays to occur after an activity instance, i.e., _ex-post_ configuration (`TimerPlacement.AFTER`), or to occur
  before an activity
  instance, i.e., _ex-ante_ configuration (`TimerPlacement.BEFORE`).
- Enhance the BPS model with the discovered extraneous delays (`DirectEnhancer()`), or try to tune the discovered delays with a TPE
  hyper-optimization stage (`HyperOptEnhancer()`)

For a more detailed explanation of the different variants of the approach, we refer to the paper "Enhancing Business Process Simulation
Models with Extraneous Activity Delays".

## Requirements

- **Python v3.9.5+**
- **PIP v21.1.2+**
- Python dependencies: Packages listed in `requirements.txt`
- Git submodule dependencies:
    - [Prosimos](https://github.com/AutomatedProcessImprovement/Prosimos)
    - [Start Time Estimator](https://github.com/AutomatedProcessImprovement/start-time-estimator)
    - [PIX Utils](https://github.com/AutomatedProcessImprovement/pix-utils)
    - [Log Distance Measures](https://github.com/AutomatedProcessImprovement/log-distance-measures)

```shell
$ git submodule update --init --recursive
$ cd ./external_tools/
$ cd ./pix-utils/
$ pip install -e .
$ cd ../log-distance-measures/
$ pip install -e .
$ cd ../start-time-estimator/
$ pip install -e .
$ cd ../Prosimos/
$ pip install -e .
$ cd ../..
$ pip install -e .
```

## Basic Usage

Check this [test file](https://github.com/AutomatedProcessImprovement/extraneous-activity-delays/blob/main/tests/simple-running-example.py)
for a simple example of how to run the technique with Prosimos,
and [config file](https://github.com/AutomatedProcessImprovement/extraneous-activity-delays/blob/main/src/extraneous_activity_delays/config.py)
for an explanation of the configuration parameters.

More sophisticated configurations of the approach are used in the test files
for [synthetic](https://github.com/AutomatedProcessImprovement/extraneous-activity-delays/blob/main/tests/infsys-synthetic-complete-evaluation.py)
and [real-life](https://github.com/AutomatedProcessImprovement/extraneous-activity-delays/blob/main/tests/infsys-real-life-evaluation.py)
evaluations.

Here, we provide two example of the proposal:

### Using Prosimos as simulation engine with no TPE optimization stage

```python
import json

import pandas as pd
from lxml import etree

from estimate_start_times.config import DEFAULT_CSV_IDS
from extraneous_activity_delays.config import Configuration, SimulationModel, SimulationEngine
from extraneous_activity_delays.config import DiscoveryMethod, TimerPlacement, OptimizationMetric
from extraneous_activity_delays.enhance_with_delays import DirectEnhancer

# Set up default configuration
log_ids = DEFAULT_CSV_IDS
config = Configuration(
    log_ids=log_ids, process_name="prosimos-example",
    simulation_engine=SimulationEngine.PROSIMOS,
    discovery_method=DiscoveryMethod.COMPLEX,  # Eclipse-aware method
    timer_placement=TimerPlacement.BEFORE,  # ex-ante configuration
    optimization_metric=OptimizationMetric.RELATIVE_EMD
    # working_schedules=working_schedules  # Use this to consider resource unavailability
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
# Read simulation parameters
with open("path_to_bps_parameters.json") as json_file:
    simulation_parameters = json.load(json_file)
simulation_model = SimulationModel(bpmn_model, simulation_parameters)
# Enhance with hyper-parametrized activity delays with hold-out
enhancer = DirectEnhancer(event_log, simulation_model, config)
enhanced_simulation_model = enhancer.enhance_simulation_model_with_delays()
# Write enhanced BPS model (BPMN and parameters)
enhanced_simulation_model.bpmn_document.bpmn_document.write("path_of_enhanced_bps_model.bpmn", pretty_print=True)
with open("path_to_enhanced_bps_parameters.json") as json_file:
    json.dump(enhanced_simulation_model.simulation_parameters, json_file)
```

### Using QBP as simulation engine with TPE optimization stage

```python
import pandas as pd
from lxml import etree

from estimate_start_times.config import DEFAULT_CSV_IDS
from extraneous_activity_delays.config import Configuration, SimulationModel, SimulationEngine
from extraneous_activity_delays.config import DiscoveryMethod, TimerPlacement, OptimizationMetric
from extraneous_activity_delays.enhance_with_delays import HyperOptEnhancer

# Set up default configuration
log_ids = DEFAULT_CSV_IDS
config = Configuration(
    log_ids=log_ids, process_name="qbp-example",
    max_alpha=10.0, training_partition_ratio=0.5,
    num_iterations=100, simulation_engine=SimulationEngine.QBP,
    discovery_method=DiscoveryMethod.COMPLEX,  # Eclipse-aware method
    timer_placement=TimerPlacement.BEFORE,  # ex-ante configuration
    optimization_metric=OptimizationMetric.RELATIVE_EMD
    # working_schedules=working_schedules  # Use this to consider resource unavailability
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
simulation_model = SimulationModel(bpmn_model)
# Enhance with hyper-parametrized activity delays with hold-out
enhancer = HyperOptEnhancer(event_log, simulation_model, config)
enhanced_simulation_model = enhancer.enhance_simulation_model_with_delays()
# Write enhanced BPS model
enhanced_simulation_model.bpmn_document.write("path_of_enhanced_bps_model.bpmn", pretty_print=True)
```

### Resource unavailability: working schedules format

```python
from pix_framework.calendar.resource_calendar import RCalendar

weekly_calendars = [
    {
        "resource_name": "Jonathan",
        "time_periods": [
            {
                "from": "MONDAY",
                "to": "FRIDAY",
                "beginTime": "09:00:00.000",
                "endTime": "14:00:00.000"
            },
            {
                "from": "MONDAY",
                "to": "THURSDAY",
                "beginTime": "16:00:00.000",
                "endTime": "19:00:00.000"
            }
        ]
    }, {
        "resource_name": "DIO",
        "time_periods": [
            {
                "from": "MONDAY",
                "to": "SUNDAY",
                "beginTime": "08:00:00.000",
                "endTime": "20:00:00.000"
            }
        ]
    }
]


def from_weekly_calendar() -> dict:
    # Read calendars
    resource_calendars = {}
    for calendar in weekly_calendars:
        resource_name = calendar['resource_name']
        r_calendar = RCalendar("calendar_{}".format(resource_name))
        for slot in calendar["time_periods"]:
            r_calendar.add_calendar_item(
                slot["from"], slot["to"], slot["beginTime"], slot["endTime"]
            )
        resource_calendars[resource_name] = r_calendar
    # Return resource calendars
    return resource_calendars
```