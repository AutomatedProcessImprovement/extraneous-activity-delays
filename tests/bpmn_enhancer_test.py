from lxml import etree

from extraneous_activity_delays.bpmn_enhancer import add_timers_to_bpmn_model
from extraneous_activity_delays.config import DurationDistribution


def test_enhance_bpmn_model_with_delays():
    # Read BPMN model
    parser = etree.XMLParser(remove_blank_text=True)
    document = etree.parse("./tests/assets/timer-events-test.bpmn", parser)
    # Enhance
    timers = {
        'A': DurationDistribution(mean="60"),
        'B': DurationDistribution(mean="600"),
        'C': DurationDistribution(mean="3600"),
        'D': DurationDistribution(mean="7200"),
        'E': DurationDistribution(mean="36000")
    }
    add_timers_to_bpmn_model(document, timers)
    model = document.getroot()
    namespace = model.nsmap
    process = model.find("process", namespace)
    sim_elements = process.find("extensionElements/qbp:processSimulationInfo/qbp:elements", namespace)
    for activity in timers:
        task = process.find("task[@name='{}']".format(activity), namespace)
        timer_to_task = process.find("sequenceFlow[@id='{}']".format(task.find("incoming", namespace).text), namespace)
        timer = process.find("intermediateCatchEvent[@id='{}']".format(timer_to_task.attrib['sourceRef']), namespace)
        to_timer = process.find("sequenceFlow[@targetRef='{}']".format(timer.attrib['id']), namespace)
        # Structural asserts
        assert to_timer.attrib['targetRef'] == timer.attrib['id']
        assert to_timer.attrib['id'] == timer.find("incoming", namespace).text
        assert timer.find("outgoing", namespace).text == timer_to_task.attrib['id']
        assert timer.find("timerEventDefinition", namespace).attrib['id'] is not None
        assert timer.attrib['id'] == timer_to_task.attrib['sourceRef']
        assert timer_to_task.attrib['targetRef'] == task.attrib['id']
        assert timer_to_task.attrib['id'] == task.find("incoming", namespace).text
        # Simulation parameters asserts
        sim_timer = sim_elements.find("qbp:element[@elementId='{}']".format(timer.attrib['id']), namespace)
        duration_distribution = sim_timer.find("qbp:durationDistribution", namespace)
        time_unit = duration_distribution.find("qbp:timeUnit", namespace)
        assert time_unit.text == timers[activity].unit
        assert duration_distribution.attrib['type'] == timers[activity].type
        assert duration_distribution.attrib['mean'] == timers[activity].mean
        assert duration_distribution.attrib['arg1'] == timers[activity].arg1
        assert duration_distribution.attrib['arg2'] == timers[activity].arg2
        assert duration_distribution.attrib['rawMean'] == timers[activity].rawMean
        assert duration_distribution.attrib['rawArg1'] == timers[activity].rawArg1
        assert duration_distribution.attrib['rawArg2'] == timers[activity].rawArg2
