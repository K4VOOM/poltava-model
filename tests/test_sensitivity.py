import pandas as pd
import pytest

from sensitivity import correlation_sensitivity


def test_correlation_sensitivity_detects_perfect_positive_relationship():
    table = pd.DataFrame({
        "S0": [1, 2, 3, 4],
        "R0": [2, 1, 1, 2],
        "Ls": [1, 3, 3, 1],
        "Lr": [4, 2, 2, 4],
        "alpha_beta": [2, 4, 6, 8],
    })

    result = correlation_sensitivity(table)

    assert result["parameter"].tolist() == ["S0", "R0", "Ls", "Lr"]
    assert result["correlation"].iloc[0] == pytest.approx(1)
    assert result["correlation"].iloc[1] == pytest.approx(0, abs=1e-12)
    assert result["correlation"].iloc[2] == pytest.approx(0, abs=1e-12)
    assert result["correlation"].iloc[3] == pytest.approx(0, abs=1e-12)


def test_correlation_sensitivity_detects_perfect_negative_relationship():
    table = pd.DataFrame({
        "S0": [2, 1, 1, 2],
        "R0": [1, 2, 3, 4],
        "Ls": [1, 3, 3, 1],
        "Lr": [4, 2, 2, 4],
        "alpha_beta": [-1, -2, -3, -4],
    })

    result = correlation_sensitivity(table)

    assert result.set_index("parameter").loc["R0", "correlation"] == pytest.approx(-1)
    assert result.set_index("parameter").loc["S0", "correlation"] == pytest.approx(0, abs=1e-12)


def test_correlation_sensitivity_returns_zero_for_independent_columns():
    table = pd.DataFrame({
        "S0": [1, 2, 3, 4],
        "R0": [2, 1, 1, 2],
        "Ls": [1, 3, 3, 1],
        "Lr": [4, 2, 2, 4],
        "alpha_beta": [1, 2, 3, 4],
    })

    result = correlation_sensitivity(table).set_index("parameter")["correlation"]

    assert result["R0"] == pytest.approx(0, abs=1e-12)
    assert result["Ls"] == pytest.approx(0, abs=1e-12)
    assert result["Lr"] == pytest.approx(0, abs=1e-12)


def test_correlation_sensitivity_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing columns"):
        correlation_sensitivity(pd.DataFrame({"S0": [1, 2]}))
