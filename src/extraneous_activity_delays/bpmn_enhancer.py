import uuid

from lxml import etree
from lxml.etree import QName, ElementTree

from extraneous_activity_delays.config import DurationDistribution


def enhance_bpmn_model_with_delays(document: ElementTree, timers: dict):
    """
    Enhance the BPMN model received by adding a timer previous to each activity denoted by [timers].

    :param document: XML document containing the BPMN model to enhance.
    :param timers: dict with the name of each activity as key, and the timer configuration as value.
    """
    # Extract process
    model = document.getroot()
    namespace = model.nsmap
    process = model.find("process", namespace)
    # Extract simulation parameters
    sim_elements = process.find("extensionElements/qbp:processSimulationInfo/qbp:elements", namespace)
    if sim_elements is None:
        sim_elements = model.find("qbp:processSimulationInfo/qbp:elements", namespace)
    # Add a timer for each task
    for task in process.findall("task", namespace):
        task_name = task.attrib['name']
        if task_name in timers:
            # The activity has a prepared timer -> add it!
            task_id = task.attrib['id']
            # Create a timer to add it preceding to the task
            timer_id = "Event_{}".format(str(uuid.uuid4()))
            timer = etree.Element(
                QName(namespace[None], "intermediateCatchEvent"),
                {'id': timer_id},
                namespace
            )
            # Redirect the edge incoming to the task so it points to the timer
            edge = process.find("sequenceFlow[@targetRef='{}']".format(task_id), namespace)
            edge.attrib['targetRef'] = timer_id
            # Create edge from the timer to the task
            flow_id = "Flow_{}".format(str(uuid.uuid4()))
            flow = etree.Element(
                QName(namespace[None], "sequenceFlow"),
                {'id': flow_id, 'sourceRef': timer_id, 'targetRef': task_id},
                namespace
            )
            # Update incoming flow information inside the task
            task_incoming = task.find("incoming", namespace)
            if task_incoming is not None:
                task_incoming.text = flow_id
            # Add incoming element inside timer
            timer_incoming = etree.Element(
                QName(namespace[None], "incoming"),
                {},
                namespace
            )
            timer_incoming.text = edge.attrib['id']
            timer.append(timer_incoming)
            # Add outgoing element inside timer
            timer_outgoing = etree.Element(
                QName(namespace[None], "outgoing"),
                {},
                namespace
            )
            timer_outgoing.text = flow_id
            timer.append(timer_outgoing)
            # Timer definition element
            timer_definition_id = "TimerEventDefinition_{}".format(str(uuid.uuid4()))
            timer_definition = etree.Element(
                QName(namespace[None], "timerEventDefinition"),
                {'id': timer_definition_id},
                namespace
            )
            timer.append(timer_definition)
            # Add the elements to the process
            process.append(timer)
            process.append(flow)
            # Add the simulation config for the timer
            duration_distribution = timers[task_name]
            sim_timer = _get_simulation_timer(timer_id, duration_distribution, namespace)
            sim_elements.append(sim_timer)
    # Remove visualizing data
    visualization_element = model.find("bpmndi:BPMNDiagram", namespace)
    if visualization_element is not None:
        model.remove(visualization_element)


def _get_simulation_timer(timer_id: str, duration_distribution: DurationDistribution, namespace: dict) -> etree.Element:
    sim_timer = etree.Element(QName(namespace['qbp'], "element"), {'elementId': timer_id}, namespace)
    duration_params = {
        'type': duration_distribution.type,
        'mean': duration_distribution.mean,
        'arg1': duration_distribution.arg1,
        'arg2': duration_distribution.arg2,
        'rawMean': duration_distribution.rawMean,
        'rawArg1': duration_distribution.rawArg1,
        'rawArg2': duration_distribution.rawArg2
    }
    sim_timer_duration = etree.Element(QName(namespace['qbp'], "durationDistribution"), duration_params, namespace)
    sim_timer_unit = etree.Element(QName(namespace['qbp'], "timeUnit"), {}, namespace)
    sim_timer_unit.text = duration_distribution.unit
    sim_timer_duration.append(sim_timer_unit)
    sim_timer.append(sim_timer_duration)
    return sim_timer
