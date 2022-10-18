"""Module containing tools for model comparison"""
from typing import Dict

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_squared_error,
    r2_score,
)


def predict_outcomes(
    model: pm.Model, trace: az.InferenceData, data: pd.DataFrame
):
    """Predicts outcomes for each datapoint based on the posterior predictive of the model.

    Parameters
    ----------
    model: Model
        PyMC model to sample from.
    trace: InferenceData
        Trace containing the posterior samples.
    data: DataFrame
        Data you want to get predictions for.

    Returns
    -------
    ndarray of shape (n_observations,)
        Vector of all predictions.
    """
    # Setting model data for prediction
    with model:
        pm.set_data(
            {
                "diagnosis": data.diagnosis,
                "visit": data.visit,
                "mother_mlu": data.MOT_MLU,
                "child_id": data.child_id,
            },
            # coords={
            # "n_obs": np.arange(len(data.index)),
            # },
        )
        # Sampling the model's posterior predictive
        posterior_predictive = pm.sample_posterior_predictive(trace)
        # Calculating predictions from the mean of the posterior predictive
        predictions = posterior_predictive.posterior_predictive.y.mean(
            ["chain", "draw"]
        ).to_numpy()
    return predictions


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluates regression predictions with multiple metrics.

    Parameters
    ----------
    y_true: ndarray of shape (n_observations, )
        True outcome values.
    y_pred: ndarray of shape (n_observations, )
        Predicted outcome values.

    Returns
    -------
    dict of str to float
        Mapping from metric names to values.
    """
    return {
        "R²": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "Explained Variance": explained_variance_score(y_true, y_pred),
        "Maximum Residual Error": max_error(y_true, y_pred),
    }  # type: ignore


def compare_predictions(
    y_true: np.ndarray, predictions: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Compares predictions of different regression models on different metrics.
    Adds the mean point estimator to the mix for chance level in each metric.

    Parameters
    ----------
    y_true: ndarray of shape (n_observations, )
        True outcome values.
    predictions: dict of str to ndarray of shape (n_observations, )
        Mapping of model names to their predictions.

    Returns
    -------
    DataFrame
        Dataframe comparing the different models on multiple metrics.
    """
    (n_observations,) = y_true.shape
    mean_estimator = np.full(shape=n_observations, fill_value=y_true.mean())
    records = [
        {"Model": "Mean Point Estimator", **evaluate(y_true, mean_estimator)}
    ]
    for model_name, y_pred in predictions.items():
        records.append({"Model": model_name, **evaluate(y_true, y_pred)})
    comparison = pd.DataFrame.from_records(records)
    comparison = comparison.sort_values(by="MSE", ascending=False)
    comparison = comparison.set_index("Model")
    return comparison
