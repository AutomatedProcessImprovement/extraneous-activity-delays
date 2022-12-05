from bpdfr_simulation_engine.probability_distributions import best_fit_distribution


def infer_distribution(data: list, bins: int = 50) -> dict:
    # Call to Prosimos distribution estimator
    return best_fit_distribution(data, bins)


def scale_distribution(distribution: dict, alpha: float) -> dict:
    # TODO Scale each of the parameters of the distribution
    return distribution
