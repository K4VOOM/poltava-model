import math
from types import SimpleNamespace

import pytest
import lanchester
from lanchester import rhs, run_battle

def test_run_battle_s_wins():
    result = run_battle(100, 80, 0.5, 0.3)
    assert result["winner"] == "S"
    assert result["remaining"] == pytest.approx(78.5, abs=0.1)
    assert result["time"] == pytest.approx(1.85, abs=0.01)

def test_run_battle_r_wins_mirror_case():
    result = run_battle(80, 100, 0.3, 0.5)
    assert result["winner"] == "R"
    assert result["remaining"] == pytest.approx(78.5, abs=0.1)
    assert result["time"] == pytest.approx(1.85, abs=0.01)

def test_run_battle_draw():
    result = run_battle(100, 100, 0.5, 0.5)
    assert result["winner"] == "draw"


def test_artillery_zero_preserves_original_result():
    original = run_battle(100, 80, 0.5, 0.3)
    with_zero_artillery = run_battle(100, 80, 0.5, 0.3, artillery_S=0, artillery_R=0)

    assert with_zero_artillery == original


def test_rhs_includes_artillery_fire_as_additional_losses():
    assert rhs(0, [100, 80], 0.5, 0.3, artillery_S=2, artillery_R=1) == [-25, -52]


def test_artillery_can_change_battle_and_return_trajectory():
    result = run_battle(
        100, 80, 0.5, 0.3, artillery_S=2, artillery_R=100, return_trajectory=True
    )

    assert result["winner"] == "R"
    assert result["trajectory"]["time"][0] == 0
    assert result["trajectory"]["S"][0] == 100
    assert result["trajectory"]["R"][0] == 80
    assert result["trajectory"]["time"][-1] == pytest.approx(result["time"])
    assert result["trajectory"]["S"][-1] == pytest.approx(result["S_end"])
    assert result["trajectory"]["R"][-1] == pytest.approx(result["R_end"])


def test_artillery_rates_must_be_finite_and_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        run_battle(100, 80, 0.5, 0.3, artillery_S=-1)
    with pytest.raises(ValueError, match="finite"):
        run_battle(100, 80, 0.5, 0.3, artillery_R=math.inf)


def test_battle_extends_time_horizon_if_no_stop_event_fires(monkeypatch):
    solve_ivp = lanchester.solve_ivp
    time_spans = []

    def miss_first_horizon(*args, **kwargs):
        time_spans.append(args[1][1])
        if len(time_spans) == 1:
            return SimpleNamespace(status=0, message="time horizon reached")
        return solve_ivp(*args, **kwargs)

    monkeypatch.setattr(lanchester, "solve_ivp", miss_first_horizon)
    result = run_battle(100, 80, 0.5, 0.3, artillery_S=2, artillery_R=1)

    assert len(time_spans) == 2
    assert time_spans[1] == 2 * time_spans[0]
    assert result["time"] > 0


def test_run_battle_raises_when_side_not_above_threshold():
    with pytest.raises(ValueError, match="greater than threshold"):
        run_battle(1.0, 80, 0.5, 0.3, threshold=1.0)

@pytest.mark.parametrize("S0, R0, alpha, beta, threshold",
    [(100, 80, 0.5, 0.3, 1.0),(80, 100, 0.3, 0.5, 1.0),(50, 30, 1.0, 2.0, 1.0),(100, 80, 0.5, 0.3, 5.0)])
def test_run_battle_winner_remaining_matches_invariant(S0, R0, alpha, beta, threshold):
    result = run_battle(S0, R0, alpha, beta, threshold)
    Q = alpha * S0**2 - beta * R0**2
    if Q > 0:
        expected = math.sqrt(Q / alpha)
        shift_bound = (beta * threshold**2 / alpha) / (2 * expected)
    else:
        expected = math.sqrt(-Q / beta)
        shift_bound = (alpha * threshold**2 / beta) / (2 * expected)
    assert result["remaining"] == pytest.approx(expected, abs=shift_bound + 1e-4)


def test_run_battle_stop_fraction_rout():
    S0, R0, alpha, beta, f = 17000, 42660, 1.5e-5, 1e-5, 0.35
    result = run_battle(S0, R0, alpha, beta, stop_fraction=f)

    S_end = f * S0
    # conserved quantity: alpha*S0^2 - beta*R0^2 = alpha*S_end^2 - beta*R_end^2
    R_end = math.sqrt((alpha * S_end**2 - (alpha * S0**2 - beta * R0**2)) / beta)

    assert result["winner"] == "R", "R should win when S is routed"
    assert result["S_end"] == pytest.approx(S_end, rel=1e-3), "S should stop at 35% of S0"
    assert result["R_end"] == pytest.approx(R_end, rel=1e-3), "R_end should follow from the invariant"


def test_run_battle_stop_fraction_invalid():
    with pytest.raises(ValueError):
        run_battle(100, 80, 0.5, 0.3, stop_fraction=1.5)