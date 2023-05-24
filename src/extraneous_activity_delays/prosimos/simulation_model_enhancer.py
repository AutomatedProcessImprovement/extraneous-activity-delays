import copy

from lxml.etree import ElementTree

from extraneous_activity_delays.config import SimulationModel, TimerPlacement
from extraneous_activity_delays.utils.bpmn_enhancement import add_timer_to_bpmn_model


def add_timers_to_simulation_model(
    simulation_model: SimulationModel,
    timers: dict,
    timer_placement: TimerPlacement = TimerPlacement.BEFORE,
) -> SimulationModel:
    """
    Enhance the BPMN model received by adding a timer previous to each activity denoted by [timers].

    :param simulation_model:    SimulationModel instance with the XML document containing the BPMN model to enhance.
    :param timers:              Dict with the name of each activity as key, and the timer configuration as value.
    :param timer_placement:     Option to consider the placement of the timers either BEFORE (the extraneous delay is considered to be
                                happening previously to an activity instance) or AFTER (the extraneous delay is considered to be
                                happening afterward an activity instance) each activity.

    :return a copy of [document] enhanced with the timers in [timers].
    """
    # Extract process
    enhanced_document = copy.deepcopy(simulation_model.bpmn_document)
    model, process, namespace = _get_basic_bpmn_elements(enhanced_document)
    # Add a timer for each task
    json_timers = []
    for task in process.findall("task", namespace):
        task_name = task.attrib["name"]
        if task_name in timers:
            # The activity has a prepared timer -> add it!
            timer_id = add_timer_to_bpmn_model(task, process, namespace, timer_placement=timer_placement)
            # Add the simulation config for the timer
            duration_distribution = timers[task_name].to_prosimos_distribution()
            json_timers += [{"event_id": timer_id} | duration_distribution]
    # Add timers to simulation parameters
    enhanced_parameters = simulation_model.simulation_parameters | {"event_distribution": json_timers}
    # Return enhanced document
    return SimulationModel(enhanced_document, enhanced_parameters)


def _get_basic_bpmn_elements(document: ElementTree) -> tuple:
    model = document.getroot()
    namespace = model.nsmap
    process = model.find("process", namespace)
    # Return elements
    return model, process, namespace
