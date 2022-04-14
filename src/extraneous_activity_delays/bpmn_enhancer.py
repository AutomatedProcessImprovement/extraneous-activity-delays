import uuid

from lxml import etree
from lxml.etree import QName

from extraneous_activity_delays.config import DurationDistribution


def enhance_bpmn_model_with_delays(input_path: str):
    # Read BPMN model
    parser = etree.XMLParser(remove_blank_text=True)
    document = etree.parse(input_path, parser)
    # Extract process and simulation parameters
    model = document.getroot()
    namespace = model.nsmap
    process = model.find("process", namespace)
    sim_elements = process.find("extensionElements/qbp:processSimulationInfo/qbp:elements", namespace)
    # Add a timer for each task
    for task in process.findall("task", namespace):
        task_id = task.attrib['id']
        # Create a timer to add it before the task
        timer_id = "Event_{}".format(str(uuid.uuid4()))
        timer = etree.Element("intermediateCatchEvent", {'id': timer_id}, namespace)
        # Create flow from timer to task
        flow_id = "Flow_{}".format(str(uuid.uuid4()))
        flow = etree.Element("sequenceFlow", {'id': flow_id, 'sourceRef': timer_id, 'targetRef': task_id}, namespace)
        # Update incoming flow information inside the task
        task.find("incoming", namespace).text = flow_id
        # Redirect all task-incoming edges to the timer
        for edge in process.findall("sequenceFlow[@targetRef='{}']".format(task_id), namespace):
            edge.attrib['targetRef'] = timer_id
            # Incoming element inside timer
            timer_incoming = etree.Element("incoming", {}, namespace)
            timer_incoming.text = edge.attrib['id']
            timer.append(timer_incoming)
        # Outgoing element inside timer
        timer_outgoing = etree.Element("outgoing", {}, namespace)
        timer_outgoing.text = task_id
        timer.append(timer_outgoing)
        # Timer definition element
        timer_definition_id = "TimerEventDefinition_{}".format(str(uuid.uuid4()))
        timer_definition = etree.Element("timerEventDefinition", {'id': timer_definition_id}, namespace)
        timer.append(timer_definition)
        # Add the elements to the process
        process.append(timer)
        process.append(flow)
        # Add the simulation config for the timer
        duration_distribution = DurationDistribution()  # Default one
        sim_timer = get_simulation_timer(timer_id, duration_distribution, namespace)
        sim_elements.append(sim_timer)
    # Export the enhanced BPMN model
    document.write("assets/timer-events-example_output.bpmn", pretty_print=True)


def get_simulation_timer(timer_id: str, duration_distribution: DurationDistribution, namespace: dict) -> etree.Element:
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
