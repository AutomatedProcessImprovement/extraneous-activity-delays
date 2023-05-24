import uuid

from lxml import etree
from lxml.etree import QName

from extraneous_activity_delays.config import TimerPlacement


def add_timer_to_bpmn_model(task, process, namespace, timer_placement: TimerPlacement = TimerPlacement.BEFORE) -> str:
    # The activity has a prepared timer -> add it!
    task_id = task.attrib["id"]
    # Create a timer to add
    timer_id = "Event_{}".format(str(uuid.uuid4()))
    timer = etree.Element(QName(namespace[None], "intermediateCatchEvent"), {"id": timer_id}, namespace)
    # Redirect the edge incoming/outgoing to the task, so it points to the timer
    if timer_placement == TimerPlacement.BEFORE:  # Incoming edge
        edge = process.find("sequenceFlow[@targetRef='{}']".format(task_id), namespace)
        edge.attrib["targetRef"] = timer_id
        # Create edge from the timer to the task
        flow_id = "Flow_{}".format(str(uuid.uuid4()))
        flow = etree.Element(
            QName(namespace[None], "sequenceFlow"),
            {"id": flow_id, "sourceRef": timer_id, "targetRef": task_id},
            namespace,
        )
        # Update incoming flow information inside the task
        task_incoming = task.find("incoming", namespace)
        if task_incoming is not None:
            task_incoming.text = flow_id
        # Add incoming element inside timer
        timer_incoming = etree.Element(QName(namespace[None], "incoming"), {}, namespace)
        timer_incoming.text = edge.attrib["id"]
        # Add outgoing element inside timer
        timer_outgoing = etree.Element(QName(namespace[None], "outgoing"), {}, namespace)
        timer_outgoing.text = flow_id
    else:  # Outgoing edge
        edge = process.find("sequenceFlow[@sourceRef='{}']".format(task_id), namespace)
        edge.attrib["sourceRef"] = timer_id
        # Create edge from the task to the timer
        flow_id = "Flow_{}".format(str(uuid.uuid4()))
        flow = etree.Element(
            QName(namespace[None], "sequenceFlow"),
            {"id": flow_id, "sourceRef": task_id, "targetRef": timer_id},
            namespace,
        )
        # Update outgoing flow information inside the task
        task_outgoing = task.find("outgoing", namespace)
        if task_outgoing is not None:
            task_outgoing.text = flow_id
        # Add outgoing element inside timer
        timer_outgoing = etree.Element(QName(namespace[None], "outgoing"), {}, namespace)
        timer_outgoing.text = edge.attrib["id"]
        # Add incoming element inside timer
        timer_incoming = etree.Element(QName(namespace[None], "incoming"), {}, namespace)
        timer_incoming.text = flow_id
    # Append timer incoming and outgoing info
    timer.append(timer_incoming)
    timer.append(timer_outgoing)
    # Timer definition element
    timer_definition_id = "TimerEventDefinition_{}".format(str(uuid.uuid4()))
    timer_definition = etree.Element(
        QName(namespace[None], "timerEventDefinition"),
        {"id": timer_definition_id},
        namespace,
    )
    timer.append(timer_definition)
    # Add the elements to the process
    process.append(timer)
    process.append(flow)
    # Return ID of the created timer
    return timer_id
