import pandas as pd


CALIBRATION_INPUTS = ("S0", "R0", "Ls", "Lr")


def correlation_sensitivity(table, inputs=CALIBRATION_INPUTS, target="alpha_beta"):
    """Return Pearson correlations between each calibration input and alpha/beta."""
    required = [*inputs, target]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Missing columns for sensitivity analysis: {', '.join(missing)}")
    if len(table) < 2:
        raise ValueError("At least two Monte Carlo rows are required")

    correlations = table.loc[:, list(inputs) + [target]].corr(method="pearson")[target]
    return pd.DataFrame({
        "parameter": list(inputs),
        "correlation": [float(correlations[name]) for name in inputs],
    })
