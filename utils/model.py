import numpy as np
import pandas as pd
import pymc as pm


def pooled_model(data: pd.DataFrame) -> pm.Model:
    """Creates inference model based on the supplied data frame."""
    coords = {
        "n_child": data.child_id.unique(),
        "diagnosis_levels": data.diagnosis.unique(),
        "n_obs": np.arange(len(data.index)),
    }
    with pm.Model(coords=coords) as model:
        diagnosis = pm.ConstantData("diagnosis", data.diagnosis, dims="n_obs")
        visit = pm.ConstantData("visit", data.visit)
        child_id = pm.ConstantData("child_id", data.child_id)
        # Hyperiors for the intercept
        mu_i = pm.Normal("mu_intercept", mu=0.3, sigma=0.2)
        sigma_i = pm.HalfNormal("sigma_intercept", 0.2)
        # Hyperiors for the slope
        mu_s = pm.Normal("mu_slope", mu=0.2, sigma=0.3)
        sigma_s = pm.HalfNormal("sigma_slope", 0.4)
        # Prior of the error
        error = pm.HalfNormal("error", 0.5)
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
            intercept[child_id, diagnosis]
            + (visit - 1) * slope[child_id, diagnosis]
        )
        # Declaring y to be LogNormally distributed around the regression line
        y = pm.LogNormal("y", mu=mu_y, sigma=error, observed=data.child_mlu)
    return model


def unpooled_model(data: pd.DataFrame) -> pm.Model:
    """Creates inference model based on the supplied data frame."""
    coords = {
        "n_child": data.child_id.unique(),
        "diagnosis_levels": data.diagnosis.unique(),
        "n_obs": np.arange(len(data.index)),
    }
    with pm.Model(coords=coords) as model:
        diagnosis = pm.ConstantData("diagnosis", data.diagnosis, dims="n_obs")
        visit = pm.ConstantData("visit", data.visit)
        child_id = pm.ConstantData("child_id", data.child_id)
        # Hyperiors for the intercept
        mu_i = pm.Normal(
            "mu_intercept", mu=0.3, sigma=0.3, dims=("diagnosis_levels")
        )
        sigma_i = pm.HalfNormal(
            "sigma_intercept", 0.3, dims=("diagnosis_levels")
        )
        # Hyperiors for the slope
        mu_s = pm.Normal(
            "mu_slope", mu=0.2, sigma=0.1, dims=("diagnosis_levels")
        )
        sigma_s = pm.HalfNormal("sigma_slope", 0.1, dims=("diagnosis_levels"))
        # Prior of the error
        error = pm.HalfNormal("error", 0.3)
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
            intercept[child_id, diagnosis]
            + (visit - 1) * slope[child_id, diagnosis]
        )
        # Declaring y to be LogNormally distributed around the regression line
        y = pm.LogNormal("y", mu=mu_y, sigma=error, observed=data.child_mlu)
    return model
