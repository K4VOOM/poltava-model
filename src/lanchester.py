import numpy as np
from scipy.integrate import solve_ivp


def rhs(t, y, alpha, beta):
    S, R = y
    return [-beta * R, -alpha * S]


def run_battle(S0, R0, alpha, beta, threshold=1.0, stop_fraction=None):
    """Lanchester square-law battle.

    Stopping rule:
      * stop_fraction is None: runs until one side drops to `threshold` people;
      * stop_fraction = f (0 < f < 1): runs until one side drops to f * its
        initial size (rout threshold).

    Returns a dict:
      winner    : "S", "R" or "draw"
      remaining : survivors on the winning side (on a draw, the level at stop time)
      time      : time at which the battle ended
      S_end, R_end : strengths of both sides at the end of the battle
    """
    if stop_fraction is None:
        S_stop = R_stop = threshold
    else:
        if not 0 < stop_fraction < 1:
            raise ValueError("stop_fraction must be between 0 and 1")
        S_stop, R_stop = stop_fraction * S0, stop_fraction * R0

    if S0 <= S_stop or R0 <= R_stop:
        raise ValueError("S0 and R0 must be greater than threshold")

    # solve_ivp passes `args` to events too, so they need the same signature as rhs
    def s_dead(t, y, alpha, beta):
        return y[0] - S_stop

    def r_dead(t, y, alpha, beta):
        return y[1] - R_stop

    for ev in (s_dead, r_dead):
        ev.terminal = True    # stop integrating when the event fires
        ev.direction = -1     # fire only when the value crosses zero going down

    # Solution is A*e^(kt) + B*e^(-kt), k = sqrt(alpha*beta). A draw ends at
    # ln(S0/S_stop)/k; the extra 40/k covers near-draws, where a tiny growing
    # mode delays the end (imbalance down to ~e^-40, below double precision).
    k = np.sqrt(alpha * beta)
    t_max = (np.log(max(S0 / S_stop, R0 / R_stop)) + 40) / k

    sol = solve_ivp(
        rhs, (0, t_max), [S0, R0],
        args=(alpha, beta),
        events=(s_dead, r_dead),
        rtol=1e-8, atol=1e-8,
    )

    # status 1 means a terminal event fired; anything else is not a finished battle
    if sol.status != 1:
        raise RuntimeError(f"Battle did not finish: {sol.message}")

    S_end, R_end = sol.y[:, -1]
    t_end = sol.t[-1]

    # Winner: the side whose stopping event did NOT fire. A draw is only
    # possible when the conserved quantity alpha*S^2 - beta*R^2 is (almost) 0.
    Q = alpha * S0**2 - beta * R0**2
    if abs(Q) <= 1e-9 * (alpha * S0**2 + beta * R0**2):
        winner, remaining = "draw", max(S_end, R_end)
    elif len(sol.t_events[0]) > 0:      # S hit its stop level -> R won
        winner, remaining = "R", R_end
    else:                               # R hit its stop level -> S won
        winner, remaining = "S", S_end

    return {
        "winner": winner,
        "remaining": float(remaining),
        "time": float(t_end),
        "S_end": float(S_end),
        "R_end": float(R_end),
    }