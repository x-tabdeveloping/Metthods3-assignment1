"""Module containing model definitions."""
import numpy as np
import pandas as pd
import pymc as pm


def unpooled_model(data: pd.DataFrame) -> pm.Model:
    """Creates inference model based on the supplied data frame."""
    coords = {
        "n_child": data.child_id.unique(),
        "diagnosis_levels": data.diagnosis.unique(),
        # "n_obs": np.arange(len(data.index)),
        "n_visit": np.arange(6),
    }
    with pm.Model() as model:
        for name, value in coords.items():
            model.add_coord(name, value, mutable=True)
        diagnosis = pm.MutableData(
            "diagnosis", data.diagnosis
        )  # , dims="n_obs")
        visit = pm.MutableData("visit", data.visit)
        mother_mlu = pm.MutableData(
            "mother_mlu", data.MOT_MLU
        )  # , dims="n_obs")
        child_id = pm.MutableData("child_id", data.child_id)
        mu_i = pm.Normal(
            "mu_intercept", mu=0.3, sigma=0.3, dims=("diagnosis_levels")
        )
        sigma_i = pm.HalfNormal(
            "sigma_intercept", 0.3, dims=("diagnosis_levels")
        )
        mu_s = pm.Normal(
            "mu_slope", mu=0.2, sigma=0.05, dims=("diagnosis_levels")
        )
        sigma_s = pm.HalfNormal(
            "sigma_slope",
            0.02,
            dims=("diagnosis_levels"),
        )
        # Prior of the error
        error = pm.HalfNormal(
            "error",
            0.02,
        )
        # Priors for the intercept drawn from the hyperior
        intercept = pm.Normal(
            "intercept",
            mu=mu_i,
            sigma=sigma_i,
            dims=("n_child", "diagnosis_levels"),
        )
        # Prior for the slopes of drawn from the hyperior
        slope = pm.Normal(
            "slope",
            mu=mu_s,
            sigma=sigma_s,
            dims=("n_child", "diagnosis_levels"),
        )
        # Expected value of MLU
        mu_y = (
            intercept[child_id, diagnosis] + visit * slope[child_id, diagnosis]
        )
        # Declaring y to be LogNormally distributed around the regression line
        y = pm.LogNormal(
            "y",
            mu=mu_y,
            sigma=error,
            observed=data.child_mlu,
            shape=diagnosis.shape,  # type: ignore
        )
    return model


def alternative_model(
    data: pd.DataFrame,
    add_mother_mlu: bool = False,
    add_verbal_iq: bool = False,
) -> pm.Model:
    """Creates inference model based on the supplied data frame."""
    # Dropping empty rows if necessary
    if add_mother_mlu:
        data = data.dropna(axis="index", subset=["MOT_MLU"])
    if add_verbal_iq:
        data = data.dropna(axis="index", subset=["verbalIQ1"])
    coords = {
        "n_child": data.child_id.unique(),
        "diagnosis_levels": data.diagnosis.unique(),
        # "n_obs": np.arange(len(data.index)),
        "n_visit": np.arange(6),
    }
    with pm.Model() as model:
        for name, value in coords.items():
            model.add_coord(name, value, mutable=True)
        diagnosis = pm.MutableData(
            "diagnosis", data.diagnosis
        )  # , dims="n_obs")
        mother_mlu = pm.MutableData(
            "mother_mlu", data.MOT_MLU
        )  # , dims="n_obs")
        visit = pm.MutableData("visit", data.visit)
        child_id = pm.MutableData("child_id", data.child_id)
        mu_i = pm.TruncatedNormal(
            "mu_intercept",
            mu=1.5,
            sigma=0.5,
            lower=0,
            dims=("diagnosis_levels"),
        )
        sigma_i = pm.HalfNormal(
            "sigma_intercept", 0.5, dims=("diagnosis_levels")
        )
        mu_s = pm.Normal(
            "mu_slope", mu=0.0, sigma=0.5, dims=("diagnosis_levels")
        )
        sigma_s = pm.HalfNormal(
            "sigma_slope",
            0.4,
            dims=("diagnosis_levels"),
        )
        # Prior of the error
        error = pm.HalfNormal(
            "error",
            0.3,
            # dims=("n_visit")
        )
        # Priors for the intercept drawn from the hyperprior
        intercept = pm.Normal(
            "intercept",
            mu=mu_i,
            sigma=sigma_i,
            dims=("n_child", "diagnosis_levels"),
        )
        # Prior for the slopes of drawn from the hyperprior
        slope = pm.Normal(
            "slope",
            mu=mu_s,
            sigma=sigma_s,
            dims=("n_child", "diagnosis_levels"),
        )
        # Expected value of MLU
        mu_y = (
            intercept[child_id, diagnosis] + visit * slope[child_id, diagnosis]
        )
        if add_mother_mlu:
            # Prior for the slope of mother mlu
            slope_mot_mlu = pm.Normal("slope_mot_mlu", mu=0.0, sigma=1.0)
            mu_y += slope_mot_mlu * mother_mlu
        if add_verbal_iq:
            verbal_iq = pm.MutableData(
                "verbal_iq", data.verbalIQ1  # , dims="n_obs"
            )
            # Prior for the slope of verbal iq
            slope_verbal_iq = pm.Normal("slope_verbal_iq", mu=0.0, sigma=1.0)
            mu_y += slope_verbal_iq * verbal_iq
        # Declaring y to be LogNormally distributed around the regression line
        y = pm.TruncatedNormal(
            "y",
            mu=mu_y,
            sigma=error,  # [visit - 1],
            lower=0.0,
            observed=data.child_mlu,
            shape=diagnosis.shape,  # type: ignore
        )
    return model
