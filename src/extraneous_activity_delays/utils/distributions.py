import math
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np
import scipy.stats as st
from scipy.stats import wasserstein_distance


@dataclass
class DurationDistribution:
    name: str = "fix"  # supported 'fix', 'expon', 'norm', 'uniform', 'lognorm', and 'gamma'
    mean: float = 0.0
    var: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0


def get_best_distribution(data: list) -> DurationDistribution:
    # TODO Remove outliers
    # TODO Add fix as a candidate distribution to measure with EMD
    # Check for fixed value
    fix_value = check_fix(data)
    if fix_value is not None:
        # If it is a fixed value, infer distribution
        distribution = DurationDistribution("fix", fix_value, 0.0, 0.0, fix_value, fix_value)
    else:
        # Otherwise, compute basic statistics and try with other distributions
        mean = np.mean(data)
        var = np.var(data)
        std = np.std(data)
        d_min = min(data)
        d_max = max(data)
        # Create distribution candidates
        dist_candidates = [
            DurationDistribution("expon", mean, var, std, d_min, d_max),
            DurationDistribution("norm", mean, var, std, d_min, d_max),
            DurationDistribution("uniform", mean, var, std, d_min, d_max)
        ]
        if mean != 0:
            dist_candidates += [DurationDistribution("lognorm", mean, var, std, d_min, d_max)]
            if var != 0:
                dist_candidates += [DurationDistribution("gamma", mean, var, std, d_min, d_max)]
        # Search for the best one within the candidates
        best_distribution = None
        best_emd = sys.float_info.max
        for distribution_candidate in dist_candidates:
            # Generate a list of observations from the distribution
            generated_data = _generate_sample(distribution_candidate, len(data))
            # Compute its distance with the observed data
            emd = wasserstein_distance(data, generated_data)
            # Update best distribution if better
            if emd < best_emd:
                best_emd = emd
                best_distribution = distribution_candidate
        # Set the best distribution as the one to return
        distribution = best_distribution
    # Return best distribution
    return distribution


def check_fix(data_list, delta=5):
    value = None
    counter = Counter(data_list)
    counter[None] = 0
    for d1 in counter:
        if (counter[d1] > counter[value]) and (sum([abs(d1 - d2) < delta for d2 in data_list]) / len(data_list) > 0.95):
            # If the value [d1] is more frequent than the current fixed one [value]
            # and
            # the ratio of values similar (or with a difference lower than [delta]) to [d1] is more than 90%
            # update value
            value = d1
    # Return fixed value with more apparitions
    return value


def _generate_sample(distribution: DurationDistribution, size: int) -> list:
    sample = []
    if distribution.name == "fix":
        sample = [distribution.mean] * size
    elif distribution.name == "expon":
        # 'loc' displaces the samples, a loc=100 will be the same than adding 100 to each sample taken from a loc=1
        sample = st.expon.rvs(loc=distribution.min, scale=distribution.mean - distribution.min, size=size)
    elif distribution.name == "norm":
        sample = st.norm.rvs(loc=distribution.mean, scale=distribution.std, size=size)
    elif distribution.name == "uniform":
        sample = st.uniform.rvs(loc=distribution.min, scale=distribution.max - distribution.min, size=size)
    elif distribution.name == "lognorm":
        # If the distribution corresponds to a 'lognorm' with loc!=0, the estimation is done wrong
        # dunno how to take that into account
        pow_mean = pow(distribution.mean, 2)
        phi = math.sqrt(distribution.var + pow_mean)
        mu = math.log(pow_mean / phi)
        sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
        sample = st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=size)
    elif distribution.name == "gamma":
        # If the distribution corresponds to a 'gamma' with loc!=0, the estimation is done wrong
        # dunno how to take that into account
        sample = st.gamma.rvs(pow(distribution.mean, 2) / distribution.var, loc=0, scale=distribution.var / distribution.mean, size=size)
    # Return generated sample
    return sample


def scale_distribution(distribution: DurationDistribution, alpha: float) -> DurationDistribution:
    return DurationDistribution(
        name=distribution.name,
        mean=distribution.mean * alpha,  # Mean: scaled by multiplying by [alpha]
        var=distribution.var * alpha * alpha,  # Variance: scaled by multiplying by [alpha]^2
        std=distribution.std * alpha,  # STD: scaled by multiplying by [alpha]
        min=distribution.min * alpha,  # Min: scaled by multiplying by [alpha]
        max=distribution.max * alpha  # Max: scaled by multiplying by [alpha]
    )
