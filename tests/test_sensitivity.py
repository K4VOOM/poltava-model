import pandas as pd
import pytest

from sensitivity import correlation_sensitivity


def test_correlation_sensitivity_returns_each_input_correlation():
    table = pd.DataFrame({
        "S0": [1, 2, 3],
        "R0": [3, 2, 1],
        "Ls": [1, 1, 1],
        "Lr": [2, 3, 4],
        "alpha_beta": [1, 2, 3],
    })

    result = correlation_sensitivity(table)

    assert result["parameter"].tolist() == ["S0", "R0", "Ls", "Lr"]
    assert result["correlation"].iloc[0] == pytest.approx(1)
    assert result["correlation"].iloc[1] == pytest.approx(-1)
    assert pd.isna(result["correlation"].iloc[2])
    assert result["correlation"].iloc[3] == pytest.approx(1)


def test_correlation_sensitivity_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing columns"):
        correlation_sensitivity(pd.DataFrame({"S0": [1, 2]}))
