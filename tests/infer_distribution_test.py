import scipy.stats as st

from extraneous_activity_delays.infer_distribution import infer_distribution


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
    assert abs(float(distribution.mean) - 100) < 2
    assert abs(float(distribution.arg1) - 20) < 2


def test_infer_distribution_exponential():
    distribution = st.expon(loc=2, scale=700)
    data = distribution.rvs(size=1000)
    distribution = infer_distribution(data)
    assert distribution.type == "EXPONENTIAL"
    assert abs(float(distribution.mean) - 3) < 1
    assert abs(float(distribution.arg1) - 700) < 50


def test_infer_distribution_uniform():
    distribution = st.uniform(loc=600, scale=120)
    data = distribution.rvs(size=1000)
    distribution = infer_distribution(data)
    assert distribution.type == "UNIFORM"
    assert abs(float(distribution.mean) - 600) < 2
    assert abs(float(distribution.arg1) - 120) < 2


def test_infer_distribution_triangular():
    distribution = st.triang(c=0.7, loc=600, scale=1000)
    data = distribution.rvs(size=1000)
    distribution = infer_distribution(data)
    assert distribution.type == "TRIANGULAR"
    assert abs(float(distribution.mean) - 0.7) < 0.05
    assert abs(float(distribution.arg1) - 600) < 30
    assert abs(float(distribution.arg2) - 1000) < 30


def test_infer_distribution_log_normal():
    distribution = st.lognorm(s=0.5, loc=600, scale=300)
    data = distribution.rvs(size=1000)
    distribution = infer_distribution(data)
    assert distribution.type == "LOGNORMAL"
    assert abs(float(distribution.mean) - 0.5) < 0.1
    assert abs(float(distribution.arg1) - 600) < 50
    assert abs(float(distribution.arg2) - 300) < 50


def test_infer_distribution_gamma():
    distribution = st.gamma(a=0.7, loc=600, scale=300)
    data = distribution.rvs(size=1000)
    distribution = infer_distribution(data)
    assert distribution.type == "GAMMA"
    assert abs(float(distribution.mean) - 0.7) < 0.15
    assert abs(float(distribution.arg1) - 600) < 20
    assert abs(float(distribution.arg2) - 300) < 100
