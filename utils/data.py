"""Module containing tools for data loading and manipulation"""
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


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepares data for analysis."""
    data = data.rename(
        columns={
            "CHI_MLU": "child_mlu",
            "Diagnosis": "diagnosis",
            "Visit": "visit",
            "Child.ID": "child_id",
            "Gender": "gender",
        }
    )
    # Mapping TD and ASD to integers
    mapping = pd.Series({"TD": 0, "ASD": 1})
    data = data.assign(diagnosis=data.diagnosis.map(mapping))
    # Factorizing children
    id_values, id_uniques = pd.factorize(data.child_id)
    data = data.assign(child_id=id_values)
    # Substracting 1 from visit, so that it starts from zero
    data = data.assign(visit=data.visit - 1)
    return data


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filters invalid observations from the dataset."""
    # Selecting necessary fields, and dropping rows containing NAs
    data = data.dropna(
        axis="index", subset=["child_mlu", "visit", "diagnosis", "child_id"]
    )
    data = data[data.child_mlu != 0]
    return data
