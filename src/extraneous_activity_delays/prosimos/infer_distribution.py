import math

from extraneous_activity_delays.utils.distributions import DurationDistribution


def parse_duration_distribution(distribution: DurationDistribution) -> dict:
    # Initialize empty list of params
    distribution_params = []
    # Add specific params depending on distribution
    if distribution.name == "fix":
        distribution_params += [
            {'value': distribution.mean}  # fixed value
        ]
    elif distribution.name == "expon":
        distribution_params += [
            {'value': distribution.min},  # loc
            {'value': distribution.mean - distribution.min},  # scale
            {'value': distribution.min},  # min
            {'value': distribution.max}  # max
        ]
    elif distribution.name == "norm":
        distribution_params += [
            {'value': distribution.mean},  # loc
            {'value': distribution.std},  # scale
            {'value': distribution.min},  # min
            {'value': distribution.max}  # max
        ]
    elif distribution.name == "uniform":
        distribution_params += [
            {'value': distribution.min},  # loc
            {'value': distribution.max - distribution.min},  # scale
            {'value': distribution.min},  # min
            {'value': distribution.max}  # max
        ]
    elif distribution.name == "lognorm":
        # If the distribution corresponds to a 'lognorm' with loc!=0, the estimation is done wrong
        # dunno how to take that into account
        pow_mean = pow(distribution.mean, 2)
        phi = math.sqrt(distribution.var + pow_mean)
        mu = math.log(pow_mean / phi)
        sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
        distribution_params += [
            {'value': sigma},  # sigma
            {'value': 0},  # loc
            {'value': math.exp(mu)},  # scale
            {'value': distribution.min},  # min
            {'value': distribution.max}  # max
        ]
    elif distribution.name == "gamma":
        # If the distribution corresponds to a 'gamma' with loc!=0, the estimation is done wrong
        # dunno how to take that into account
        distribution_params += [
            {'value': pow(distribution.mean, 2) / distribution.var},  # a
            {'value': 0},  # loc
            {'value': distribution.var / distribution.mean},  # scale
            {'value': distribution.min},  # min
            {'value': distribution.max}  # max
        ]
    # Return dict with the distribution data as expected by PROSIMOS
    return {'distribution_name': distribution.name, 'distribution_params': distribution_params}
