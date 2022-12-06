import uuid

from lxml import etree
from lxml.etree import QName


def add_timer_to_bpmn_model(task, process, namespace) -> str:
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
    # Return ID of the created timer
    return timer_id
