import numpy as np
import scipy.stats as st

from extraneous_activity_delays.config import QBPDurationDistribution
from extraneous_activity_delays.qbp.infer_distribution import infer_distribution, scale_distribution


def test_infer_distribution_fixed():
    data = [150] * 1000
    distribution = infer_distribution(data)
    assert distribution.type == "FIXED"
    assert distribution.mean == "150"


def test_infer_distribution_fixed_with_noise():
    data = [149] * 100 + [150] * 1000 + [151] * 100 + [200] * 5
    distribution = infer_distribution(data)
    assert distribution.type == "FIXED"
    assert distribution.mean == "150"


def test_infer_distribution_not_fixed():
    data = [147] * 28 + [150] * 26 + [151] * 32 + [240] * 14
    distribution = infer_distribution(data)
    assert distribution.type != "FIXED"


def test_infer_distribution_normal():
    distribution = st.norm(loc=100, scale=20)
    data = distribution.rvs(size=1000)
    distribution = infer_distribution(data)
    assert distribution.type == "NORMAL"
    assert distribution.mean == str(np.mean(data))
    assert distribution.arg1 == str(np.std(data))
    assert distribution.arg2 == "0"


def test_infer_distribution_exponential():
    distribution = st.expon(loc=2, scale=700)
    data = distribution.rvs(size=1000)
    distribution = infer_distribution(data)
    assert distribution.type == "EXPONENTIAL"
    assert distribution.mean == "0"
    assert distribution.arg1 == str(np.mean(data))
    assert distribution.arg2 == "0"


def test_infer_distribution_uniform():
    distribution = st.uniform(loc=600, scale=120)
    data = distribution.rvs(size=1000)
    distribution = infer_distribution(data)
    assert distribution.type == "UNIFORM"
    assert distribution.mean == "3600"
    assert distribution.arg1 == str(np.min(data))
    assert distribution.arg2 == str(np.max(data))


def test_infer_distribution_triangular():
    distribution = st.triang(c=0.7, loc=600, scale=1000)
    data = distribution.rvs(size=1000)
    distribution = infer_distribution(data)
    assert distribution.type == "TRIANGULAR"
    assert distribution.mean == str(st.mode([int(seconds) for seconds in data]).mode[0])
    assert distribution.arg1 == str(np.min(data))
    assert distribution.arg2 == str(np.max(data))


def test_infer_distribution_log_normal():
    distribution = st.lognorm(s=0.5, loc=600, scale=300)
    data = distribution.rvs(size=1000)
    distribution = infer_distribution(data)
    assert distribution.type == "LOGNORMAL"
    assert distribution.mean == str(np.mean(data))
    assert distribution.arg1 == str(np.var(data))
    assert distribution.arg2 == "0"


def test_infer_distribution_gamma():
    distribution = st.gamma(a=0.7, loc=600, scale=300)
    data = distribution.rvs(size=1000)
    distribution = infer_distribution(data)
    assert distribution.type == "GAMMA"
    assert distribution.mean == str(np.mean(data))
    assert distribution.arg1 == str(np.var(data))
    assert distribution.arg2 == "0"


def test_scale_distribution_log_fixed():
    alpha = 0.85
    distribution = QBPDurationDistribution("FIXED", "176", "0", "0")
    scaled = scale_distribution(distribution, alpha)
    assert scaled.type == "FIXED"
    assert scaled.mean == str(176 * alpha)
    assert scaled.arg1 == "0"
    assert scaled.arg2 == "0"


def test_scale_distribution_normal():
    alpha = 0.1
    distribution = QBPDurationDistribution("NORMAL", "10", "2", "0")
    scaled = scale_distribution(distribution, alpha)
    assert scaled.type == "NORMAL"
    assert scaled.mean == str(10 * alpha)
    assert scaled.arg1 == str(2 * alpha)
    assert scaled.arg2 == "0"


def test_scale_distribution_exponential():
    alpha = 0.9
    distribution = QBPDurationDistribution("EXPONENTIAL", "0", "10", "0")
    scaled = scale_distribution(distribution, alpha)
    assert scaled.type == "EXPONENTIAL"
    assert scaled.mean == "0"
    assert scaled.arg1 == str(10 * alpha)
    assert scaled.arg2 == "0"


def test_scale_distribution_uniform():
    alpha = 0.75
    distribution = QBPDurationDistribution("UNIFORM", "3600", "2", "12")
    scaled = scale_distribution(distribution, alpha)
    assert scaled.type == "UNIFORM"
    assert scaled.mean == "3600"
    assert scaled.arg1 == str(2 * alpha)
    assert scaled.arg2 == str(12 * alpha)


def test_scale_distribution_triangular():
    alpha = 0.25
    distribution = QBPDurationDistribution("TRIANGULAR", "10", "2", "18")
    scaled = scale_distribution(distribution, alpha)
    assert scaled.type == "TRIANGULAR"
    assert scaled.mean == str(10 * alpha)
    assert scaled.arg1 == str(2 * alpha)
    assert scaled.arg2 == str(18 * alpha)


def test_scale_distribution_log_normal():
    alpha = 0.6
    distribution = QBPDurationDistribution("LOGNORMAL", "10", "2", "0")
    scaled = scale_distribution(distribution, alpha)
    assert scaled.type == "LOGNORMAL"
    assert scaled.mean == str(10 * alpha)
    assert scaled.arg1 == str(2 * alpha * alpha)
    assert scaled.arg2 == "0"


def test_scale_distribution_log_gamma():
    alpha = 0.5
    distribution = QBPDurationDistribution("GAMMA", "11", "5", "0")
    scaled = scale_distribution(distribution, alpha)
    assert scaled.type == "GAMMA"
    assert scaled.mean == str(11 * alpha)
    assert scaled.arg1 == str(5 * alpha * alpha)
    assert scaled.arg2 == "0"
