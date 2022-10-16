from typing import Dict

import arviz as az
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def add_abline(
    fig: go.Figure, a: float, b: float, color: str = "black", **kwargs
) -> go.Figure:
    scatter_data, *_ = fig.data
    x = scatter_data["x"]  # type: ignore
    x0: float = min(x)  # type: ignore
    x1: float = max(x)  # type: ignore
    y0, y1 = a * x0 + b, a * x1 + b
    trace = go.Scatter(
        name="",
        x=[x0, x1],
        y=[y0, y1],
        mode="lines",
        showlegend=False,
        line=dict(color=color),
        **kwargs,
    )
    return fig.add_trace(trace)


def plot_each(plots: Dict[str, go.Figure]):
    for var, plot in plots.items():
        print(var)
        plot.show()


def save_each(plots: Dict[str, go.Figure], prefix: str):
    for var, plot in plots.items():
        plot.write_image(f"plots/{prefix}_{var}.png")


def plot_prior_posterior_update(
    trace: az.InferenceData,
) -> Dict[str, go.Figure]:
    """Plots visualization of how the parameters get updated by the data.

    Parameters
    ----------
    trace: InferenceData
        Trace containing sample from the prior and the posterior.

    Note
    ----
    The amount of samples from the posterior and prior HAS TO be the same for this to work
    properly.
    Please make sure that you extend the trace with the proper amount of prior samples.
    """
    posterior = (
        trace.posterior.to_dataframe()  # type: ignore
        .reset_index()
        .drop(columns=["n_child", "chain", "draw"])
    ).assign(group="posterior")
    prior = (
        trace.prior.to_dataframe()  # type: ignore
        .reset_index()
        .drop(columns=["n_child", "chain", "draw"])
    ).assign(group="prior")
    inference_df = pd.concat((posterior, prior), ignore_index=True)
    if len(posterior.index) > 200_000:
        inference_df = inference_df.sample(200_000)
    inference_df = inference_df.melt(id_vars=["group", "diagnosis_levels"])
    variables = trace.posterior.data_vars.keys()
    plots = dict()
    for var in variables:
        data = inference_df[inference_df.variable == var]
        plot = (
            px.histogram(
                data,
                title=var,
                facet_col="diagnosis_levels",
                x="value",
                color="group",
                barmode="overlay",
                height=400,
            )
            .update_yaxes(matches=None)
            .update_xaxes(matches=None)
        )
        plots[var] = plot
    return plots


def plot_posterior_group_comparison(
    trace: az.InferenceData,
) -> Dict[str, go.Figure]:
    """Plots the posterior parameters for both groups overlayed.

    Parameters
    ----------
    trace: InferenceData
        Trace containing sample from the posterior.
    """
    posterior = (
        trace.posterior.to_dataframe()
        .reset_index()
        .drop(columns=["n_child", "chain", "draw"])
        .melt(id_vars=["diagnosis_levels"])
    )
    if len(posterior.index) > 200_000:
        posterior = posterior.sample(200_000)
    variables = trace.posterior.data_vars.keys()
    plots = dict()
    for var in variables:
        data = posterior[posterior.variable == var]
        plot = (
            px.histogram(
                data,
                title=var,
                x="value",
                color="diagnosis_levels",
                barmode="overlay",
                height=400,
            )
            .update_yaxes(matches=None)
            .update_xaxes(matches=None)
        )
        plots[var] = plot
    return plots


def plot_power_analysis(data: pd.DataFrame) -> go.Figure:
    """Plots the power analysis data with error bars.

    Parameters
    ----------
    data: DataFrame
        Data obtained from the power analysis.

    Returns
    -------
    Figure
        Scatter plot displaying median difference in slopes in the estimates
        as well as error bars representing 95% credible intervals(HDI).
    """
    return px.scatter(
        data,
        x="trial",
        y="median_difference",
        error_y="error",
        error_y_minus="error_minus",
        facet_col="group_size",
        facet_col_wrap=2,
    ).add_hline(y=0)
