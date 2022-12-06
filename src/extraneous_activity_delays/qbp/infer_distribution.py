from dataclasses import dataclass

from extraneous_activity_delays.utils.distributions import DurationDistribution


@dataclass
class QBPDurationDistribution:
    type: str = "NORMAL"
    mean: str = "NaN"  # Warning! these values are always interpreted as seconds
    arg1: str = "NaN"
    arg2: str = "NaN"
    unit: str = "seconds"  # This is the unit to show in the interface by transforming the values in seconds


def parse_duration_distribution(distribution: DurationDistribution) -> QBPDurationDistribution:
    # Initialize empty distribution
    qbp_distribution = None
    # Parse distribution
    if distribution.name == 'fix':
        qbp_distribution = QBPDurationDistribution(
            type="FIXED",
            mean=str(distribution.mean),
            arg1="0",
            arg2="0"
        )
    elif distribution.name == 'expon':
        # For the XML mean=0 and arg2=0
        qbp_distribution = QBPDurationDistribution(
            type="EXPONENTIAL",
            mean="0",
            arg1=str(distribution.mean),
            arg2="0"
        )
    elif distribution.name == 'norm':
        # For the XML arg1=std and arg2=0
        qbp_distribution = QBPDurationDistribution(
            type="NORMAL",
            mean=str(distribution.mean),
            arg1=str(distribution.std),
            arg2="0"
        )
    elif distribution.name == 'uniform':
        # For the XML the mean is always 3600, arg1=min and arg2=max
        qbp_distribution = QBPDurationDistribution(
            type="UNIFORM",
            mean="3600",
            arg1=str(distribution.min),
            arg2=str(distribution.max)
        )
    elif distribution.name == 'lognorm':
        # For the XML arg1=var and arg2=0
        qbp_distribution = QBPDurationDistribution(
            type="LOGNORMAL",
            mean=str(distribution.mean),
            arg1=str(distribution.var),
            arg2="0"
        )
    elif distribution.name == 'gamma':
        # For the XML arg1=var and arg2=0
        qbp_distribution = QBPDurationDistribution(
            type="GAMMA",
            mean=str(distribution.mean),
            arg1=str(distribution.var),
            arg2="0"
        )
    # Return parsed distribution
    return qbp_distribution
