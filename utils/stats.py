"""Module containing statistical utilities for bayesian inference"""

from typing import Tuple
import arviz as az
import numpy as np
from numpy.typing import ArrayLike


def credible_interval(a: ArrayLike) -> Tuple[float, float]:
    """Computes 95% credible interval of a given set of data with HDI estimation.

    Parameters
    ----------
    a: array-like
        Sequence containing the data.

    Returns
    -------
    low: float
        Lower boundary of the credible interval.
    high: float
        Upper boundary of the credible interval.
    """
    a = np.array(a)
    # low, high = np.quantile(a, 0.025), np.quantile(a, 0.975)
    low, high = az.hdi(a, hdi_prob=0.95)
    return low, high  # type: ignore
