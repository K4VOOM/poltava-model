import numpy as np


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