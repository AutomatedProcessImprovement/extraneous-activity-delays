import sys
import warnings
from collections import Counter

import numpy as np
import scipy.stats as st

# Create models from data
from extraneous_activity_delays.config import DurationDistribution


def infer_distribution(data: list, bins: int = 50) -> DurationDistribution:
    duration_distribution = None
    # Check if the value is fixed and no distribution is needed
    fix_value = check_fix(data)
    if fix_value is not None:
        # Save fixed value
        duration_distribution = DurationDistribution(type="FIXED", mean=str(fix_value))
    else:
        # Model data by finding best fit distribution to data
        # Get histogram of original data
        d_min = min(data + [sys.float_info.max])
        d_max = max(data + [0])
        y, x = np.histogram(data, bins=bins, density=True)
        # Transform the start and end of each bin, in its middle point
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Possible distributions: BIMP supported
        distributions = [st.norm, st.expon, st.uniform, st.triang, st.lognorm, st.gamma]
        # Initialize search with normal distribution
        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf

        # Estimate distribution parameters from data
        for distribution in distributions:
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    # Fit distribution to data
                    params = distribution.fit(data)
                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))
                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse
            except Exception:
                pass
        duration_distribution = _parse_duration_distribution(best_distribution, data)
    # Return best duration distribution
    return duration_distribution


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


def _parse_duration_distribution(distribution, data) -> DurationDistribution:
    if distribution == st.norm:
        # For the XML arg1=std and arg2=0
        return DurationDistribution(
            type="NORMAL",
            mean=str(np.mean(data)),
            arg1=str(np.std(data)),
            arg2="0"
        )
    elif distribution == st.expon:
        # For the XML mean=0 and arg2=0
        return DurationDistribution(
            type="EXPONENTIAL",
            mean="0",
            arg1=str(np.mean(data)),
            arg2="0"
        )
    elif distribution == st.uniform:
        # For the XML the mean is always 3600, arg1=min and arg2=max
        return DurationDistribution(
            type="UNIFORM",
            mean="3600",
            arg1=str(np.min(data)),
            arg2=str(np.max(data))
        )
    elif distribution == st.triang:
        # For the XML the mode is stored in the mean parameter, arg1=min and arg2=max
        return DurationDistribution(
            type="TRIANGULAR",
            mean=str(st.mode([int(seconds) for seconds in data]).mode[0]),
            arg1=str(np.min(data)),
            arg2=str(np.max(data))
        )
    elif distribution == st.lognorm:
        # For the XML arg1=var and arg2=0
        return DurationDistribution(
            type="LOGNORMAL",
            mean=str(np.mean(data)),
            arg1=str(np.var(data)),
            arg2="0"
        )
    elif distribution == st.gamma:
        # For the XML arg1=var and arg2=0
        return DurationDistribution(
            type="GAMMA",
            mean=str(np.mean(data)),
            arg1=str(np.var(data)),
            arg2="0"
        )
