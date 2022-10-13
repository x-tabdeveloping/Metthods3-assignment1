import pandas as pd
import arviz as az


def inferencedata_to_df(inference_data: az.InferenceData) -> pd.DataFrame:
    """Turns inference data to DataFrame format, so that it's easier to display.

    Parameters
    ----------
    prior_check: InferenceData
        Arviz data structure containing the inference data.

    Returns
    -------
    DataFrame
        Data frame containing the inference data.
        Each row represents a single observation.
    """
    # Gets a mapping of constant data from the prior check
    constant_data = inference_data.constant_data.data_vars  # type: ignore
    observed = inference_data.observed_data.data_vars  # type: ignore
    if hasattr(inference_data, "prior_predictive"):
        pred = inference_data.prior_predictive  # type: ignore
    elif hasattr(inference_data, "posterior_predictive"):
        pred = inference_data.posterior_predictive  # type: ignore
    else:
        raise ValueError(
            "Inferencedata has neither of these attributes:"
            "{prior_predictive, posterior_predictive}"
        )
    predictions = {
        f"{key}_pred": data.stack(sample=("draw", "chain")).to_numpy().tolist()
        for key, data in pred.data_vars.items()
    }
    result = pd.DataFrame({**constant_data, **predictions, **observed})
    # Exploding the results of multiple draws into one column
    result = result.explode(list(predictions.keys()))  # type: ignore
    return result
