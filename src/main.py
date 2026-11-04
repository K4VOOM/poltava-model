import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibration import ratio_alpha_beta
from data_loader import load_parameters
from lanchester import run_battle
from montecarlo import (
    NARVA_CALIBRATION_NAMES,
    run_calibration_mc,
    save_results,
    summarize_alpha_beta,
)
from sensitivity import correlation_sensitivity


ROOT = Path(__file__).resolve().parent.parent


def _plot_distributions(poltava, narva, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, values, title in (
        (axes[0], poltava["alpha_beta"], "Полтава, 1709"),
        (axes[1], narva["alpha_beta"], "Нарва, 1700"),
    ):
        ax.hist(values, bins=40, density=True, alpha=0.75)
        ax.axvline(values.median(), color="darkred", linestyle="--", label=f"Медіана {values.median():.2f}")
        ax.set(title=title, xlabel="α/β", ylabel="Щільність")
        ax.legend()
    fig.suptitle("Розподіл каліброваного відношення α/β")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_sensitivity(sensitivity, path):
    labels = {"S0": "S₀: шведи", "R0": "R₀: росіяни", "Ls": "Lₛ: втрати шведів", "Lr": "Lᵣ: втрати росіян"}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([labels[name] for name in sensitivity["parameter"]], sensitivity["correlation"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Чутливість α/β до вхідних величин", ylabel="Кореляція Пірсона", ylim=(-1, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_trajectory(trajectory, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trajectory["time"], trajectory["S"], label="Шведи S")
    ax.plot(trajectory["time"], trajectory["R"], label="Росіяни R")
    ax.set(title="Траєкторія каліброваної моделі Полтавської битви", xlabel="Умовний час", ylabel="Чисельність")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _add_transfer_check(narva, poltava_ratio):
    transfer = narva.copy()
    remaining_squared = transfer["S0"] ** 2 - (
        transfer["R0"] ** 2 - transfer["R_end"] ** 2
    ) / poltava_ratio
    feasible = remaining_squared >= 0
    predicted = np.full(len(transfer), np.nan)
    predicted[feasible] = np.sqrt(remaining_squared[feasible])
    transfer["poltava_ratio_transfer_feasible"] = feasible
    transfer["S_end_predicted_from_poltava_ratio"] = predicted
    return transfer


def _summary_table(poltava, narva, poltava_ratio):
    poltava_summary = summarize_alpha_beta(poltava)
    narva_summary = summarize_alpha_beta(narva)
    feasible = narva["poltava_ratio_transfer_feasible"]
    predicted = narva.loc[feasible, "S_end_predicted_from_poltava_ratio"]
    return pd.DataFrame([
        {
            "battle": "Полтава, 1709",
            **poltava_summary,
            "multiple_vs_poltava_median": 1.0,
            "poltava_ratio_transfer_feasible_fraction": np.nan,
            "median_predicted_S_end": np.nan,
        },
        {
            "battle": "Нарва, 1700",
            **narva_summary,
            "multiple_vs_poltava_median": narva_summary["median"] / poltava_ratio,
            "poltava_ratio_transfer_feasible_fraction": float(feasible.mean()),
            "median_predicted_S_end": float(predicted.median()) if not predicted.empty else np.nan,
        },
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Калібрування моделі битв і створення графіків та таблиць.")
    parser.add_argument("--samples", type=int, default=10000, help="кількість симуляцій Монте-Карло")
    parser.add_argument("--seed", type=int, default=42, help="зерно генератора випадкових чисел")
    args = parser.parse_args(argv)
    if args.samples < 2:
        parser.error("--samples має бути не менше 2")

    figures_dir = ROOT / "figures"
    tables_dir = ROOT / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    parameters = load_parameters()
    poltava = run_calibration_mc(parameters, args.samples, args.seed)
    sensitivity = correlation_sensitivity(poltava)

    narva = run_calibration_mc(
        parameters,
        args.samples,
        args.seed + 1,
        parameter_names=NARVA_CALIBRATION_NAMES,
    )
    poltava_ratio = summarize_alpha_beta(poltava)["median"]
    narva = _add_transfer_check(narva, poltava_ratio)

    likely = {name: values[1] for name, values in parameters.items()}
    S0, R0 = likely["S0_attack"], likely["R0_engaged"]
    Ls, Lr = likely["val_Ls_killed_captured"], likely["val_Lr_killed_wounded"]
    likely_ratio = ratio_alpha_beta(S0, R0, S0 - Ls, R0 - Lr)
    trajectory_result = run_battle(
        S0,
        R0,
        alpha=likely_ratio,
        beta=1.0,
        stop_fraction=(S0 - Ls) / S0,
        return_trajectory=True,
    )

    _plot_distributions(poltava, narva, figures_dir / "alpha_beta_distribution.png")
    _plot_sensitivity(sensitivity, figures_dir / "sensitivity.png")
    _plot_trajectory(trajectory_result["trajectory"], figures_dir / "battle_trajectory.png")

    save_results(poltava, tables_dir / "poltava_mc_results.csv")
    save_results(narva, tables_dir / "narva_mc_results.csv")
    sensitivity.to_csv(tables_dir / "sensitivity.csv", index=False)
    summary = _summary_table(poltava, narva, poltava_ratio)
    summary.to_csv(tables_dir / "battle_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nТаблиці: {tables_dir.relative_to(ROOT)}/")
    print(f"Графіки: {figures_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
