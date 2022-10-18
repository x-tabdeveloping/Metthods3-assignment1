#!/usr/bin/env python
# coding: utf-8
# %%

import arviz as az
import numpy as np
import plotly.express as px
import pymc as pm
from matplotlib.pyplot import savefig, show
from pymc.sampling_jax import sample_blackjax_nuts

from utils.model import unpooled_model
from utils.plotting import (
    plot_each,
    plot_posterior_group_comparison,
    plot_power_analysis,
    plot_prior_posterior_update,
    save_each,
)
from utils.simulation import power_analysis, simulate_data

N_GROUPSIZE = 100

# %%


def sample(model: pm.Model, use_jax: bool = True) -> az.InferenceData:
    """Samples prior predictive, posterior and posterior predictive.

    Parameters
    ----------
    model: Model
        Model to sample from.
    use_jax: bool, default True
        Specifies whether JAX should be used for sampling.

    Returns
    -------
    InferenceData
        Trace with posterior, prior predictive and posterior predictive sample.
    """
    trace: az.InferenceData = pm.sample_prior_predictive(4000, model)  # type: ignore
    sampler = sample_blackjax_nuts if use_jax else pm.sample
    trace.extend(sampler(model=model))  # type: ignore
    trace.extend(pm.sample_posterior_predictive(trace, model=model))  # type: ignore
    return trace


# %%
dat = simulate_data(N_GROUPSIZE)
# %% Producing figure displaying the simulated data
fig = px.box(dat, x="visit", y="child_mlu", color="diagnosis")
fig.write_image("plots/simulation/simulated_data.png")
fig

# %% Creating model
model = unpooled_model(dat)
trace = sample(model)

# %% Plotting prior predictive
az.plot_ppc(trace, group="prior")
savefig("plots/simulation/prior_pred.png")

# %% Plotting posterior predictive
axes = az.plot_ppc(trace, group="posterior")
savefig("plots/simulation/post_pred.png")

# %% Plotting prior_posterior update
plots = plot_prior_posterior_update(trace)
save_each(plots, prefix="update", directory="simulation")
plot_each(plots)
# %% Plotting posterior trace
axes = az.plot_trace(trace)
savefig("plots/simulation/traceplot.png")

# %% Selecting random children
all_children = np.arange(N_GROUPSIZE * 2)
np.random.shuffle(all_children)
selected_children = all_children[:7]

# %% Plotting ESS evolution
az.plot_ess(
    trace, kind="evolution", coords={"n_child": selected_children.tolist()}
)
savefig("plots/simulation/ess_plot.png")

# %% Calculating summary statistics

summary = az.summary(trace)

# %% R-hat plot

fig = px.histogram(summary.r_hat).update_xaxes(type="category")
fig.write_image("plots/simulation/r_hat.png")
fig

# %% ESS bulk plot

fig = px.histogram(summary.ess_bulk)
fig.write_image("plots/simulation/ess_bulk.png")
fig

# %% ESS tail plot
fig = px.histogram(summary.ess_tail)

fig.write_image("plots/simulation/ess_tail.png")
fig

# %% Run power analysis
fit_data = power_analysis(
    inference_model=unpooled_model,
    group_sizes=[10, 80],  # , 50, 80, 100, 150],
    n_trials=5,  # 10,
    use_jax_sampler=True,
)
fit_data.to_csv("./results/power_analysis.csv")

# %% Plot power analysis
fig = plot_power_analysis(fit_data)
fig.add_hline(y=0.07, line_color="red")
fig.update_layout(height=800)
fig.write_image("plots/simulation/power_analysis.png")
fig

# %%
