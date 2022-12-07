import random

import numpy as np
import scipy.stats as st

from extraneous_activity_delays.utils.distributions import get_best_distribution, DurationDistribution, scale_distribution


def test_infer_distribution_fixed():
    data = [150] * 1000
    distribution = get_best_distribution(data)
    assert distribution.name == "fix"
    assert distribution.mean == 150.0


def test_infer_distribution_fixed_with_noise():
    data = [149] * 100 + [150] * 1000 + [151] * 100 + [200] * 5
    distribution = get_best_distribution(data)
    assert distribution.name == "fix"
    assert distribution.mean == 150.0


def test_infer_distribution_not_fixed():
    data = [147] * 28 + [150] * 26 + [151] * 32 + [240] * 14
    distribution = get_best_distribution(data)
    assert distribution.name != "fix"


def test_infer_distribution_normal():
    distribution = st.norm(loc=100, scale=20)
    data = distribution.rvs(size=1000)
    distribution = get_best_distribution(data)
    assert distribution.name == "norm"
    _assert_distribution_params(distribution, data)


def test_infer_distribution_exponential():
    distribution = st.expon(loc=2, scale=700)
    data = distribution.rvs(size=1000)
    distribution = get_best_distribution(data)
    assert distribution.name == "expon"
    _assert_distribution_params(distribution, data)


def test_infer_distribution_uniform():
    distribution = st.uniform(loc=600, scale=120)
    data = distribution.rvs(size=1000)
    distribution = get_best_distribution(data)
    assert distribution.name == "uniform"
    _assert_distribution_params(distribution, data)


def test_infer_distribution_log_normal():
    distribution = st.lognorm(s=0.5, loc=600, scale=300)
    data = distribution.rvs(size=1000)
    distribution = get_best_distribution(data)
    assert distribution.name == "lognorm"
    _assert_distribution_params(distribution, data)


def test_infer_distribution_gamma():
    distribution = st.gamma(a=0.7, loc=600, scale=300)
    data = distribution.rvs(size=1000)
    distribution = get_best_distribution(data)
    assert distribution.name == "gamma"
    _assert_distribution_params(distribution, data)


def _assert_distribution_params(distribution, data):
    assert distribution.mean == np.mean(data)
    assert distribution.var == np.var(data)
    assert distribution.std == np.std(data)
    assert distribution.min == np.min(data)
    assert distribution.max == np.max(data)


def test_scale_distributions():
    distributions = [
        DurationDistribution(name="fix", mean=60, var=0, std=0, min=60, max=60),
        DurationDistribution(name="norm", mean=1200, var=36, std=6, min=1000, max=1400),
        DurationDistribution(name="expon", mean=3600, var=100, std=10, min=1200, max=7200),
        DurationDistribution(name="uniform", mean=3600, var=4000000, std=2000, min=0, max=7200),
        DurationDistribution(name="lognorm", mean=120, var=100, std=10, min=100, max=190),
        DurationDistribution(name="gamma", mean=1200, var=144, std=12, min=800, max=1400)
    ]
    for distribution in distributions:
        alpha = random.randrange(1, 500) / 100
        scaled = scale_distribution(distribution, alpha)
        assert scaled.name == distribution.name
        assert scaled.mean == distribution.mean * alpha
        assert scaled.var == distribution.var * alpha * alpha
        assert scaled.std == distribution.std * alpha
        assert scaled.min == distribution.min * alpha
        assert scaled.max == distribution.max * alpha
