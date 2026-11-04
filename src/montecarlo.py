from pathlib import Path

import numpy as np
import pandas as pd

from calibration import ratio_alpha_beta


def sample_triangular(low, likely, high, n, rng):
    """Draw n values from a triangular distribution (min, likely, max).

    numpy refuses low == high and silently returns nan for a nan mode, so
    both cases are handled here explicitly.
    """
    if np.isnan(likely):
        raise ValueError("'likely' is missing (nan); decide its value before sampling")
    if not low <= likely <= high:
        raise ValueError("expected min <= likely <= max")
    if low == high:
        return np.full(n, float(low))
    return rng.triangular(low, likely, high, size=n)


def sample_parameters(params, n, seed, names=None):
    """Sample n values for each parameter.

    params : {name: (min, likely, max)}, e.g. from load_parameters('input')
    names  : which parameters to sample (default: all of them)
    seed   : one seed for the whole run, so the results are reproducible

    Returns {name: array of n values}. Parameters are sampled independently.
    """
    rng = np.random.default_rng(seed)
    if names is None:
        names = list(params)
    return {name: sample_triangular(*params[name], n, rng) for name in names}


CALIBRATION_NAMES = ["S0_attack", "R0_engaged", "val_Ls_killed_captured", "val_Lr_killed_wounded"]
NARVA_CALIBRATION_NAMES = ["S0_narva", "R0_narva", "val_Ls_narva", "val_Lr_narva"]


def run_calibration_mc(params, n, seed, parameter_names=CALIBRATION_NAMES):
    """Monte Carlo for the alpha/beta ratio.

    Each run draws the initial strengths (S0, R0) and the losses (Ls, Lr) from
    their triangular distributions, turns them into end-of-battle strengths and
    calibrates alpha/beta from the conserved quantity (calibration.py).
    The four inputs are drawn independently. `parameter_names` selects the
    corresponding initial-strength and loss parameters for a particular battle.

    Returns a DataFrame with n rows: S0, R0, Ls, Lr, S_end, R_end, alpha_beta.
    """
    if len(parameter_names) != 4:
        raise ValueError("parameter_names must contain S0, R0, Ls and Lr parameter names")
    s = sample_parameters(params, n, seed, names=parameter_names)

    table = pd.DataFrame({
        "S0": s[parameter_names[0]],
        "R0": s[parameter_names[1]],
        "Ls": s[parameter_names[2]],
        "Lr": s[parameter_names[3]],
    })
    table["S_end"] = table["S0"] - table["Ls"]
    table["R_end"] = table["R0"] - table["Lr"]
    table["alpha_beta"] = [
        ratio_alpha_beta(S0, R0, S_end, R_end)
        for S0, R0, S_end, R_end in zip(table["S0"], table["R0"], table["S_end"], table["R_end"])
    ]
    return table


def summarize_alpha_beta(table):
    """Median, mean and the 5th/95th percentiles of the alpha/beta column."""
    ratio = table["alpha_beta"]
    return {
        "median": float(ratio.median()),
        "mean": float(ratio.mean()),
        "p5": float(ratio.quantile(0.05)),
        "p95": float(ratio.quantile(0.95)),
    }


def save_results(table, path):
    """Write the table to a CSV file, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)