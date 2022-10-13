import io
from contextlib import redirect_stdout
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pymc as pm
from pymc.sampling_jax import sample_blackjax_nuts

from utils.stats import credible_interval


def simulate_data(group_size: int) -> pd.DataFrame:
    """Simulates data with outcomes.

    Parameters
    ----------
    group_size: int
        Number of children in each diagnosis group.

    Returns
    -------
    DataFrame
        Syntactically generated data.
    """
    asd_intercepts = np.random.normal(0.4, 0.3, size=group_size)
    td_intercepts = np.random.normal(0.4, 0.1, size=group_size)
    asd_slopes = np.random.normal(0.13, 0.02, size=group_size)
    td_slopes = np.random.normal(0.2, 0.01, size=group_size)
    children = pd.DataFrame(
        {
            "child_id": np.arange(group_size * 2),
            "diagnosis": [0] * group_size + [1] * group_size,
            "intercept": np.concatenate((td_intercepts, asd_intercepts)),
            "slope": np.concatenate((td_slopes, asd_slopes)),
        }
    )
    children = children.loc[children.index.repeat(6)]
    children = children.assign(visit=np.tile(np.arange(6) + 1, group_size * 2))
    y = np.random.lognormal(
        children.intercept + children.slope * (children.visit - 1), 0.01
    )
    children = children.assign(child_mlu=y)
    return children


def _power_analysis(
    inference_model: Callable[[pd.DataFrame], pm.Model],
    group_sizes: Iterable[int],
    n_trials: int = 10,
    n_samples: int = 1000,
    use_jax_sampler: bool = True,
) -> pd.DataFrame:
    """Fits the model on the hypothetical data with the given group sizes.

    Parameters
    ----------
    inference_model: (DataFrame) -> Model
        Function creating the model based on data.
    group_sizes: iterable of int
        Group sizes to be tested.
    n_trials: int, default 10
        Number of trials for each group size.
    n_samples: int, default 1000
        Number of samples to draw from the posterior at each trial.
    use_jax_sampler: bool, default True
        Indicates whether a JAX based sampler should be used. (blackjax)

    Returns
    -------
    DataFrame
        Credible intervals and medians of the difference of slope estimates
        drawn from the posterior.
    """
    result = pd.DataFrame(
        columns=[
            "group_size",
            "trial",
            "median_difference",
            "low",
            "high",
        ]
    )
    for group_size in group_sizes:
        print(f"Obtaining results for group size: {group_size}")
        for trial in range(n_trials):
            # Obtaining simulated sample
            data = simulate_data(group_size)
            # Creating model
            model = inference_model(data)
            print(f"    Trial no. {trial+1}/{n_trials}: sampling...")
            # Capturing stdout from sampling
            with redirect_stdout(io.StringIO()):
                if use_jax_sampler:
                    posterior = sample_blackjax_nuts(n_samples, model=model)
                else:
                    posterior = pm.sample(
                        n_samples, model=model, progressbar=False
                    )
            slope = posterior.posterior.slope.to_dataframe().reset_index()  # type: ignore
            slope = slope.pivot(
                index=("chain", "draw", "n_child"),
                columns="diagnosis_levels",
                values="slope",
            )
            slope = slope.assign(difference=slope[0] - slope[1]).drop(
                columns=[0, 1]
            )
            slope = slope.groupby(["chain", "draw"]).mean()
            slope = slope.assign(group_size=group_size, trial=trial)
            low, high = credible_interval(slope.difference)
            credibility = pd.DataFrame(
                {
                    "group_size": group_size,
                    "trial": trial,
                    "median_difference": np.median(slope.difference),
                    "low": low,
                    "high": high,
                },
                index=[0],
            )
            result = pd.concat((result, credibility), ignore_index=True)
    result = result.assign(
        error_plus=result.high - result.median_difference,
        error_minus=np.abs(result.low - result.median_difference),
    )
    return result


def power_analysis(
    inference_model: Callable[[pd.DataFrame], pm.Model],
    group_sizes: Iterable[int],
    n_trials: int = 10,
    n_samples: int = 1000,
    use_jax_sampler: bool = True,
) -> pd.DataFrame:
    """Fits the model on the hypothetical data with the given group sizes.

    Parameters
    ----------
    inference_model: (DataFrame) -> Model
        Function creating the model based on data.
    group_sizes: iterable of int
        Group sizes to be tested.
    n_trials: int, default 10
        Number of trials for each group size.
    n_samples: int, default 1000
        Number of samples to draw from the posterior at each trial.
    use_jax_sampler: bool, default True
        Indicates whether a JAX based sampler should be used. (blackjax)

    Returns
    -------
    DataFrame
        Credible intervals and medians of the difference of slope estimates
        drawn from the posterior.
    """
    records = []
    for group_size in group_sizes:
        print(f"Obtaining results for group size: {group_size}")
        for trial in range(n_trials):
            # Obtaining simulated sample
            data = simulate_data(group_size)
            # Creating model
            model = inference_model(data)
            print(f"    Trial no. {trial+1}/{n_trials}: sampling...")
            # Capturing stdout from sampling
            with redirect_stdout(io.StringIO()):
                if use_jax_sampler:
                    posterior = sample_blackjax_nuts(n_samples, model=model)
                else:
                    posterior = pm.sample(
                        n_samples, model=model, progressbar=False
                    )
            slope = posterior.posterior.mu_slope.to_dataframe().reset_index()  # type: ignore
            slope = slope.pivot(
                index=("chain", "draw"),
                columns="diagnosis_levels",
                values="mu_slope",
            )
            slope = slope.assign(difference=slope[0] - slope[1])
            slope = slope.reset_index()
            median_difference = np.median(slope.difference)
            low, high = credible_interval(slope.difference)
            records.append(
                {
                    "trial": trial,
                    "group_size": group_size,
                    "median_difference": median_difference,
                    "low": low,
                    "high": high,
                }
            )
    results = pd.DataFrame.from_records(records)
    results = results.assign(
        error=results.high - results.median_difference,
        error_minus=results.median_difference - results.low,
    )
    return results
