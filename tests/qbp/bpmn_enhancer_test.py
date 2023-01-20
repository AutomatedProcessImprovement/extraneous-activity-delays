from lxml import etree

from extraneous_activity_delays.config import SimulationModel, TimerPlacement
from extraneous_activity_delays.qbp.simulation_model_enhancer import add_timers_to_simulation_model
from extraneous_activity_delays.utils.distributions import DurationDistribution


def test_enhance_bpmn_model_with_delays():
    # Read BPMN model
    parser = etree.XMLParser(remove_blank_text=True)
    document = etree.parse("./tests/assets/timer-events-test.bpmn", parser)
    simulation_model = SimulationModel(document)
    # Enhance
    timers = {
        'A': DurationDistribution(name="fix", mean=60, var=0, std=0, min=60, max=60),
        'B': DurationDistribution(name="norm", mean=1200, var=36, std=6, min=1000, max=1400),
        'C': DurationDistribution(name="expon", mean=3600, var=100, std=10, min=1200, max=7200),
        'D': DurationDistribution(name="uniform", mean=3600, var=4000000, std=2000, min=0, max=7200),
        'E': DurationDistribution(name="gamma", mean=1200, var=144, std=12, min=800, max=1400)
    }
    enhanced_simulation_model = add_timers_to_simulation_model(simulation_model, timers)
    model = enhanced_simulation_model.bpmn_document.getroot()
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
        assert time_unit.text == "seconds"
        if timers[activity].name == "fix":
            assert duration_distribution.attrib['type'] == "FIXED"
            assert duration_distribution.attrib['mean'] == str(timers[activity].mean)
            assert duration_distribution.attrib['arg1'] == "0"
            assert duration_distribution.attrib['arg2'] == "0"
        elif timers[activity].name == "norm":
            assert duration_distribution.attrib['type'] == "NORMAL"
            assert duration_distribution.attrib['mean'] == str(timers[activity].mean)
            assert duration_distribution.attrib['arg1'] == str(timers[activity].std)
            assert duration_distribution.attrib['arg2'] == "0"
        elif timers[activity].name == "expon":
            assert duration_distribution.attrib['type'] == "EXPONENTIAL"
            assert duration_distribution.attrib['mean'] == "0"
            assert duration_distribution.attrib['arg1'] == str(timers[activity].mean)
            assert duration_distribution.attrib['arg2'] == "0"
        elif timers[activity].name == "uniform":
            assert duration_distribution.attrib['type'] == "UNIFORM"
            assert duration_distribution.attrib['mean'] == "3600"
            assert duration_distribution.attrib['arg1'] == str(timers[activity].min)
            assert duration_distribution.attrib['arg2'] == str(timers[activity].max)
        elif timers[activity].name == "gamma":
            assert duration_distribution.attrib['type'] == "GAMMA"
            assert duration_distribution.attrib['mean'] == str(timers[activity].mean)
            assert duration_distribution.attrib['arg1'] == str(timers[activity].var)
            assert duration_distribution.attrib['arg2'] == "0"
        else:
            assert False


def test_enhance_bpmn_model_with_delays_after():
    # Read BPMN model
    parser = etree.XMLParser(remove_blank_text=True)
    document = etree.parse("./assets/timer-events-test.bpmn", parser)
    simulation_model = SimulationModel(document)
    # Enhance
    timers = {
        'A': DurationDistribution(name="fix", mean=60, var=0, std=0, min=60, max=60),
        'B': DurationDistribution(name="norm", mean=1200, var=36, std=6, min=1000, max=1400),
        'C': DurationDistribution(name="expon", mean=3600, var=100, std=10, min=1200, max=7200),
        'D': DurationDistribution(name="uniform", mean=3600, var=4000000, std=2000, min=0, max=7200),
        'E': DurationDistribution(name="gamma", mean=1200, var=144, std=12, min=800, max=1400)
    }
    enhanced_simulation_model = add_timers_to_simulation_model(simulation_model, timers, TimerPlacement.AFTER)
    model = enhanced_simulation_model.bpmn_document.getroot()
    namespace = model.nsmap
    process = model.find("process", namespace)
    sim_elements = process.find("extensionElements/qbp:processSimulationInfo/qbp:elements", namespace)
    for activity in timers:
        task = process.find("task[@name='{}']".format(activity), namespace)
        task_to_timer = process.find("sequenceFlow[@id='{}']".format(task.find("outgoing", namespace).text), namespace)
        timer = process.find("intermediateCatchEvent[@id='{}']".format(task_to_timer.attrib['targetRef']), namespace)
        from_timer = process.find("sequenceFlow[@sourceRef='{}']".format(timer.attrib['id']), namespace)
        # Structural asserts
        assert from_timer.attrib['sourceRef'] == timer.attrib['id']
        assert from_timer.attrib['id'] == timer.find("outgoing", namespace).text
        assert timer.find("incoming", namespace).text == task_to_timer.attrib['id']
        assert timer.find("timerEventDefinition", namespace).attrib['id'] is not None
        assert timer.attrib['id'] == task_to_timer.attrib['targetRef']
        assert task_to_timer.attrib['sourceRef'] == task.attrib['id']
        assert task_to_timer.attrib['id'] == task.find("outgoing", namespace).text
        # Simulation parameters asserts
        sim_timer = sim_elements.find("qbp:element[@elementId='{}']".format(timer.attrib['id']), namespace)
        duration_distribution = sim_timer.find("qbp:durationDistribution", namespace)
        time_unit = duration_distribution.find("qbp:timeUnit", namespace)
        assert time_unit.text == "seconds"
        if timers[activity].name == "fix":
            assert duration_distribution.attrib['type'] == "FIXED"
            assert duration_distribution.attrib['mean'] == str(timers[activity].mean)
            assert duration_distribution.attrib['arg1'] == "0"
            assert duration_distribution.attrib['arg2'] == "0"
        elif timers[activity].name == "norm":
            assert duration_distribution.attrib['type'] == "NORMAL"
            assert duration_distribution.attrib['mean'] == str(timers[activity].mean)
            assert duration_distribution.attrib['arg1'] == str(timers[activity].std)
            assert duration_distribution.attrib['arg2'] == "0"
        elif timers[activity].name == "expon":
            assert duration_distribution.attrib['type'] == "EXPONENTIAL"
            assert duration_distribution.attrib['mean'] == "0"
            assert duration_distribution.attrib['arg1'] == str(timers[activity].mean)
            assert duration_distribution.attrib['arg2'] == "0"
        elif timers[activity].name == "uniform":
            assert duration_distribution.attrib['type'] == "UNIFORM"
            assert duration_distribution.attrib['mean'] == "3600"
            assert duration_distribution.attrib['arg1'] == str(timers[activity].min)
            assert duration_distribution.attrib['arg2'] == str(timers[activity].max)
        elif timers[activity].name == "gamma":
            assert duration_distribution.attrib['type'] == "GAMMA"
            assert duration_distribution.attrib['mean'] == str(timers[activity].mean)
            assert duration_distribution.attrib['arg1'] == str(timers[activity].var)
            assert duration_distribution.attrib['arg2'] == "0"
        else:
            assert False
