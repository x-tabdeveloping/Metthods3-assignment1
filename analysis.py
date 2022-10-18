"""Script to produce the data analysis part of the assignment."""
# %%
import arviz as az
import pandas as pd
import plotly.express as px
import pymc as pm
from matplotlib.pyplot import savefig

from utils.model import unpooled_model
from utils.plotting import save_each, plot_prior_posterior_update

# %% Loading data and renaming columns for easier usage
data = pd.read_csv("data_clean.csv")
data = data.rename(
    columns={
        "CHI_MLU": "child_mlu",
        "Diagnosis": "diagnosis",
        "Visit": "visit",
        "Child.ID": "child_id",
        "Gender": "gender",
    }
)
counts = data.groupby("diagnosis").agg(
    count=("child_id", lambda s: s.nunique())
)
print(counts)

# %% Removing problematic data
# Mapping TD and ASD to integers
mapping = pd.Series({"TD": 0, "ASD": 1})
data = data.assign(diagnosis=data.diagnosis.map(mapping))
# Selecting necessary fields, and dropping rows containing NAs
data = data.dropna(
    axis="index", subset=["child_mlu", "visit", "diagnosis", "child_id"]
)
# Factorizing children
id_values, id_uniques = pd.factorize(data.child_id)
data["child_id"] = id_values
# Removing 0 values (lognormal wouldn't like this)
data = data[data.child_mlu != 0]
new_counts = data.groupby("diagnosis").agg(
    count=("child_id", lambda s: s.nunique())
)
print(new_counts)

# %% Plotting number of participants in both groups and genders
counts = (
    data.groupby(["diagnosis", "gender"])
    .agg(count=("child_id", lambda s: s.nunique()))
    .reset_index()
)
fig = px.bar(
    counts,
    x="diagnosis",
    y="count",
    color="gender",
)
fig.write_image("plots/analysis/counts_bar.png")
fig

# %% Descriptive statistics
summary = data.groupby("diagnosis").describe(exclude=[int, "object"])

# %% Visualizing verbal IQ
fig = px.box(data, x="diagnosis", y="verbalIQ1")
print(summary["verbalIQ1"])
fig.write_image("plots/analysis/verbal_iq.png")
fig

# %% Visualizing non-verbal IQ
fig = px.box(data, x="diagnosis", y="nonVerbalIQ1")
print(summary["nonVerbalIQ1"])
fig.write_image("plots/analysis/non_verbal_iq.png")
fig

# %% Visualizing real data
fig = px.box(data, x="visit", y="child_mlu", color="diagnosis", width=700)
fig.write_image("plots/analysis/real_data.png")
fig


# %% Sampling the posterior
model = unpooled_model(data)
trace = pm.sample(1000, model=model, tune=1000, target_accept=0.99)

# %% Sampling the prior and posterior predictive
trace.extend(pm.sample_posterior_predictive(trace, model=model))
trace.extend(pm.sample_prior_predictive(4000, model=model))
trace.to_netcdf("models/default_model.nc")


# %% Prior predictive check
az.plot_ppc(trace, group="prior")
savefig("plots/analysis/prior_pred.png")

# %% Posterior predictive check
az.plot_ppc(trace, group="posterior")
savefig("plots/analysis/post_pred.png")

# %% Traceplot
az.plot_trace(trace)
savefig("plots/analysis/trace_plot.png")

# %% Prior-posterior update plots
plots = plot_prior_posterior_update(trace)
save_each(plots, prefix="update", directory="analysis")
