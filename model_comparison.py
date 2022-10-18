"""Script doing model comparison. Can also be used in VSCode's interactive mode."""
# %%
import arviz as az
import pandas as pd
import plotly.express as px
import pymc as pm
from matplotlib.pyplot import savefig, show
from utils.data import filter_data, prepare_data

from utils.model import unpooled_model, alternative_model
from utils.model_comparison import (
    compare_predictions,
    predict_outcomes,
)


def fit_save_model(model: pm.Model, name: str):
    """Samples posterior, posterior predictive and prior predictive.
    Saves trace to "/models/<name>.nc"
    """
    trace = pm.sample(model=model, target_accept=0.99)
    trace.extend(pm.sample_prior_predictive(4000, model=model))  # type: ignore
    trace.extend(pm.sample_posterior_predictive(trace, model=model))  # type: ignore
    trace.to_netcdf(f"models/{name}.nc")  # type: ignore


# %% Loading data and renaming columns for easier usage
data = pd.read_csv("data_clean.csv")
data = prepare_data(data)
data = filter_data(data)

# %% Fitting the different models on the dataset.
alt_model = alternative_model(data)
fit_save_model(alt_model, name="alt_model")
alt_model_mother = alternative_model(data, add_mother_mlu=True)
fit_save_model(alt_model_mother, name="alt_model_mother")
alt_model_mother_iq = alternative_model(
    data, add_mother_mlu=True, add_verbal_iq=True
)
fit_save_model(alt_model_mother_iq, name="alt_model_mother_iq")
alt_model_iq = alternative_model(data, add_verbal_iq=True)
fit_save_model(alt_model_iq, name="alt_model_iq")

# %% Load traces from disk
trace = az.InferenceData.from_netcdf("models/default_model.nc")
alt_trace = az.InferenceData.from_netcdf("models/alt_model.nc")
alt_trace_iq = az.InferenceData.from_netcdf("models/alt_model_iq.nc")
alt_trace_mother = az.InferenceData.from_netcdf("models/alt_model_mother.nc")
alt_trace_mother_iq = az.InferenceData.from_netcdf(
    "models/alt_model_mother_iq.nc"
)
# Creating mappings
models = {
    "Default": unpooled_model(data),
    "Alternative": alternative_model(
        data, add_mother_mlu=False, add_verbal_iq=False
    ),
    "Alternative with Verbal IQ": alternative_model(
        data, add_mother_mlu=False, add_verbal_iq=True
    ),
    "Alternative with Mother MLU": alternative_model(
        data, add_mother_mlu=True, add_verbal_iq=False
    ),
    "Alternative with Mother MLU + Verbal IQ": alternative_model(
        data, add_mother_mlu=True, add_verbal_iq=True
    ),
}
traces = {
    "Default": trace,
    "Alternative": alt_trace,
    "Alternative with Verbal IQ": alt_trace_iq,
    "Alternative with Mother MLU": alt_trace_mother,
    "Alternative with Mother MLU + Verbal IQ": alt_trace_mother_iq,
}
# %%
loos = {model_name: az.loo(trace) for model_name, trace in traces.items()}
az.plot_elpd(loos)
savefig("plots/model_comparison/loo_plot.png")
# %%
comp = (
    pd.DataFrame.from_dict(
        loos,
        orient="index",
    )
    .drop(columns=["warning", "loo_i", "pareto_k", "loo_scale"])
    .sort_values(by="loo", ascending=False)
)
comp.to_csv("results/model_comparison.csv")
comp

# %% Loading test dataset
test_data = pd.read_csv("test_clean.csv")

# %% Generating predictions for each model for the new data
# Removing those models that include verbal IQ, as we don't have
# test data for that.
testable = models.copy()
testable.pop("Alternative with Verbal IQ", None)
testable.pop("Alternative with Mother MLU + Verbal IQ", None)
predictions = {
    model_name: predict_outcomes(model, traces[model_name], test_data)
    for model_name, model in testable.items()
}
pred_comp = compare_predictions(test_data.child_mlu, predictions)
pred_comp.to_csv("results/prediction_quality.csv")
