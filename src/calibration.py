def ratio_alpha_beta(S0, R0, S_end, R_end):
    """Ratio alpha/beta from the start and end state of one battle.

    Square-law Lanchester: alpha*S^2 - beta*R^2 stays constant during the battle, so
        alpha*S0^2 - beta*R0^2 = alpha*S_end^2 - beta*R_end^2
        =>  alpha/beta = (R0^2 - R_end^2) / (S0^2 - S_end^2)

    Only the ratio is identifiable from losses; the absolute scale
    (k = sqrt(alpha*beta)) would need the duration of the battle.
    """
    if not 0 <= S_end < S0:
        raise ValueError("S_end must be in [0, S0)")
    if not 0 <= R_end < R0:
        raise ValueError("R_end must be in [0, R0)")

    return (R0**2 - R_end**2) / (S0**2 - S_end**2)