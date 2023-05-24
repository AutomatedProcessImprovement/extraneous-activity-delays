import copy
import datetime

from lxml import etree
from lxml.etree import QName, ElementTree

from extraneous_activity_delays.config import SimulationModel, TimerPlacement
from extraneous_activity_delays.utils.bpmn_enhancement import add_timer_to_bpmn_model
from pix_framework.statistics.distribution import QBPDurationDistribution


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
    model, process, sim_info, namespace = _get_basic_bpmn_elements(enhanced_document)
    sim_elements = sim_info.find("qbp:elements", namespace)
    # Add a timer for each task
    for task in process.findall("task", namespace):
        task_name = task.attrib["name"]
        if task_name in timers:
            # The activity has a prepared timer -> add it!
            timer_id = add_timer_to_bpmn_model(task, process, namespace, timer_placement=timer_placement)
            # Add the simulation config for the timer
            duration_distribution = timers[task_name].to_qbp_distribution()
            sim_timer = _get_simulation_timer(timer_id, duration_distribution, namespace)
            sim_elements.append(sim_timer)
    # Remove visualizing data
    visualization_element = model.find("bpmndi:BPMNDiagram", namespace)
    if visualization_element is not None:
        model.remove(visualization_element)
    # Return enhanced document
    return SimulationModel(enhanced_document)


def _get_simulation_timer(
    timer_id: str, duration_distribution: QBPDurationDistribution, namespace: dict
) -> etree.Element:
    sim_timer = etree.Element(QName(namespace["qbp"], "element"), {"elementId": timer_id}, namespace)
    duration_params = {
        "type": duration_distribution.type,
        "mean": duration_distribution.mean,
        "arg1": duration_distribution.arg1,
        "arg2": duration_distribution.arg2,
    }
    sim_timer_duration = etree.Element(QName(namespace["qbp"], "durationDistribution"), duration_params, namespace)
    sim_timer_unit = etree.Element(QName(namespace["qbp"], "timeUnit"), {}, namespace)
    sim_timer_unit.text = duration_distribution.unit
    sim_timer_duration.append(sim_timer_unit)
    sim_timer.append(sim_timer_duration)
    return sim_timer


def _get_basic_bpmn_elements(document: ElementTree) -> tuple:
    model = document.getroot()
    namespace = model.nsmap
    if "qbp" not in namespace:
        namespace["qbp"] = "http://www.qbp-simulator.com/Schema201212"
    process = model.find("process", namespace)
    # Extract simulation parameters
    sim_info = process.find("extensionElements/qbp:processSimulationInfo", namespace)
    if sim_info is None:
        # SIMOD exported BPMN models
        sim_info = model.find("qbp:processSimulationInfo", namespace)
    # Return elements
    return model, process, sim_info, namespace


def set_number_instances_to_simulate(document: ElementTree, num_instances: int):
    # Get basic elements
    _, _, sim_info, _ = _get_basic_bpmn_elements(document)
    # Edit num instances
    sim_info.attrib["processInstances"] = str(num_instances)


def set_start_datetime_to_simulate(document: ElementTree, time: datetime.datetime):
    # Get basic elements
    _, _, sim_info, _ = _get_basic_bpmn_elements(document)
    # Edit num instances
    sim_info.attrib["startDateTime"] = (
        time.strftime("%Y-%m-%dT%H:%M:%S.%f") + time.strftime("%z")[:-2] + ":" + time.strftime("%z")[-2:]
    )
