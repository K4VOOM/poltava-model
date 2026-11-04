import pytest

from calibration import ratio_alpha_beta
from lanchester import run_battle


def test_ratio_round_trip_with_run_battle():
    alpha, beta = 1.5e-5, 1e-5
    result = run_battle(17000, 42660, alpha, beta, stop_fraction=0.35)

    ratio = ratio_alpha_beta(17000, 42660, result["S_end"], result["R_end"])

    assert ratio == pytest.approx(alpha / beta, rel=1e-3), "ratio should match alpha/beta used in the model"


def test_ratio_valid_input_does_not_raise():
    assert ratio_alpha_beta(17000, 42660, 5950, 37940) > 0, "ratio must be positive for a valid battle"


@pytest.mark.parametrize(
    "S_end, R_end",
    [(18000, 37940), (-1, 37940), (5950, 43000), (5950, -1)],
)
def test_ratio_raises_on_invalid_end_state(S_end, R_end):
    with pytest.raises(ValueError):
        ratio_alpha_beta(17000, 42660, S_end, R_end)