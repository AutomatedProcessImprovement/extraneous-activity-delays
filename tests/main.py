from lxml import etree

from extraneous_activity_delays.bpmn_enhancer import enhance_bpmn_model_with_delays
from extraneous_activity_delays.config import DurationDistribution

if __name__ == '__main__':
    # Read BPMN model
    parser = etree.XMLParser(remove_blank_text=True)
    document = etree.parse("assets/timer-events-example.bpmn", parser)
    # Enhance
    timers = {
        'Check  application  form completeness': DurationDistribution(mean="60"),
        'Assess loan risk': DurationDistribution(mean="600"),
        'Approve application': DurationDistribution(mean="3600"),
        'Design loan offer': DurationDistribution(mean="7200"),
        'Approve Loan Offer': DurationDistribution(mean="36000")
    }
    enhanced_model = enhance_bpmn_model_with_delays(document, timers)
    # Export the enhanced BPMN model
    document.write("assets/timer-events-example_output.bpmn", pretty_print=True)
