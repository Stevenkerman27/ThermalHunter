"""Evaluate frozen dynamic-policy replicas on one held-out scenario suite."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from eval_dynamic import (
    evaluate_policy,
    load_dynamic_dqn_agent,
    load_ppo_agent,
    make_scenarios,
)
from glider_discrete_simp import RBWindField
from train_ppo import dynamic_wind_paths


def bootstrap_mean(values, seed=9173, n_resamples=20000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.mean(values)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(records):
    rows = []
    for (policy, training_seed), group in records.groupby(["policy", "training_seed"], sort=False):
        row = {
            "policy": policy,
            "training_seed": training_seed,
            "n_scenarios": len(group),
            "mean_height_change": group["height_change"].mean(),
            "median_height_change": group["height_change"].median(),
            "sd_height_change": group["height_change"].std(ddof=1),
            "mean_energy_height_change": group["energy_height_change"].mean(),
            "median_energy_height_change": group["energy_height_change"].median(),
            "sd_energy_height_change": group["energy_height_change"].std(ddof=1),
            "mean_steps": group["steps"].mean(),
        }
        _, row["height_ci_low"], row["height_ci_high"] = bootstrap_mean(
            group["height_change"], seed=9173 + int(training_seed) if isinstance(training_seed, int) else 9173
        )
        _, row["energy_ci_low"], row["energy_ci_high"] = bootstrap_mean(
            group["energy_height_change"], seed=19381 + int(training_seed) if isinstance(training_seed, int) else 19381
        )
        for reason in ("altitude_low", "altitude_high", "wind_end", "numerical_divergence"):
            row[f"termination_{reason}_fraction"] = (group["termination_reason"] == reason).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_learned(summary):
    rows = []
    learned = summary[summary["training_seed"] != "baseline"]
    for policy, group in learned.groupby("policy", sort=False):
        rows.append(
            {
                "policy": policy,
                "n_training_seeds": len(group),
                "mean_of_seed_means_height_change": group["mean_height_change"].mean(),
                "sd_across_seed_means_height_change": group["mean_height_change"].std(ddof=1),
                "min_seed_mean_height_change": group["mean_height_change"].min(),
                "max_seed_mean_height_change": group["mean_height_change"].max(),
                "mean_of_seed_means_energy_height_change": group["mean_energy_height_change"].mean(),
                "sd_across_seed_means_energy_height_change": group["mean_energy_height_change"].std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def save_figure(records, path):
    policies = ["Random grid", "Cruise", "DQN", "PPO"]
    colors = {"Random grid": "#7f8c8d", "Cruise": "#4c78a8", "DQN": "#f58518", "PPO": "#54a24b"}
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=False)
    for axis, column, ylabel in (
        (axes[0], "height_change", "Height change (m)"),
        (axes[1], "energy_height_change", "Total-energy height change (m)"),
    ):
        data = [records.loc[records["policy"] == policy, column].to_numpy() for policy in policies]
        axis.boxplot(
            data,
            positions=np.arange(len(policies)),
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "#444444"},
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#444444"},
            capprops={"color": "#444444"},
        )
        for index, policy in enumerate(policies):
            values = records.loc[records["policy"] == policy, column].to_numpy()
            # Keep scenario points visible while using x-position to identify
            # the within-policy training replica.
            for seed_index, seed in enumerate(sorted(records.loc[records["policy"] == policy, "training_seed"].unique(), key=str)):
                seed_values = records.loc[(records["policy"] == policy) & (records["training_seed"] == seed), column].to_numpy()
                jitter = np.linspace(-0.16, 0.16, len(seed_values)) if len(seed_values) > 1 else np.array([0.0])
                axis.scatter(
                    index + jitter + (seed_index - 1) * 0.012,
                    seed_values,
                    s=8,
                    alpha=0.32,
                    color=colors[policy],
                    edgecolors="none",
                    rasterized=True,
                )
        axis.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--")
        axis.set_xticks(np.arange(len(policies)), ["Random", "Cruise", "DQN", "PPO"])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
    figure.suptitle("Held-out dynamic-regime evaluation (100 matched scenarios)", fontsize=11)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    output_dir = Path(config.TRAIN_RESULT_DIR) / "dynamic_multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    seeds = tuple(config.DYNAMIC_REPLICATION_SEEDS)
    scenario_seed = config.DYNAMIC_EVAL_SEED
    scenarios = make_scenarios(config.N_EVAL_EPISODES, seed=scenario_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    wind_manager = RBWindField(dynamic_wind_paths(), memory_mode=True)
    try:
        baseline_normalizer = None
        for seed in seeds:
            ppo_path = model_dir / f"ppo_seed{seed}.pth"
            dqn_path = model_dir / f"dqn_seed{seed}.pth"
            if not ppo_path.is_file() or not dqn_path.is_file():
                raise FileNotFoundError(f"missing replica artifact for seed {seed}: {ppo_path} or {dqn_path}")
            ppo_agent, ppo_normalizer = load_ppo_agent(str(ppo_path), device)
            dqn_agent, dqn_normalizer = load_dynamic_dqn_agent(str(dqn_path), device)
            if baseline_normalizer is None:
                baseline_normalizer = ppo_normalizer
            ppo_rows = evaluate_policy(
                "PPO", scenarios, wind_manager, ppo_normalizer,
                agent=ppo_agent, device=device, scenario_seed=scenario_seed,
            )
            dqn_rows = evaluate_policy(
                "DQN", scenarios, wind_manager, dqn_normalizer,
                agent=dqn_agent, device=device, scenario_seed=scenario_seed,
            )
            for row in ppo_rows + dqn_rows:
                row["training_seed"] = seed
                row["evaluation_seed"] = scenario_seed
                records.append(row)
        for policy in ("Random grid", "Cruise"):
            baseline_rows = evaluate_policy(
                policy, scenarios, wind_manager, baseline_normalizer, scenario_seed=scenario_seed
            )
            for row in baseline_rows:
                row["training_seed"] = "baseline"
                row["evaluation_seed"] = scenario_seed
                records.append(row)
    finally:
        wind_manager.close()

    results = pd.DataFrame(records)
    results = results[[
        "policy", "training_seed", "evaluation_seed", "scenario", "height_change",
        "energy_height_change", "return", "steps", "termination_reason",
    ]].sort_values(["policy", "training_seed", "scenario"])
    summary = summarize(results)
    aggregate = aggregate_learned(summary)
    results.to_csv(output_dir / "episodes.csv", index=False)
    summary.to_csv(output_dir / "summary_by_seed.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_learned.csv", index=False)
    save_figure(results, output_dir / "dynamic_multiseed_results.png")
    protocol = {
        "evaluation_seed": scenario_seed,
        "n_scenarios": config.N_EVAL_EPISODES,
        "training_seeds": list(seeds),
        "learned_replication_unit": "training_seed",
        "scenario_unit": "matched initial condition; distributions remain within seed",
        "models": {"PPO": "continuous two-command action", "DQN": "3x3 discrete action grid"},
        "config": {
            key: getattr(config, key)
            for key in (
                "DYNAMIC_NUM_ENVS", "PPO_TOTAL_TIMESTEPS", "PPO_LEARNING_RATE", "PPO_NUM_STEPS",
                "PPO_NUM_MINIBATCHES", "PPO_UPDATE_EPOCHS", "PPO_GAMMA", "PPO_GAE_LAMBDA",
                "DYNAMIC_DQN_TOTAL_TIMESTEPS", "DYNAMIC_DQN_LEARNING_RATE", "DYNAMIC_DQN_BATCH_SIZE",
                "DYNAMIC_DQN_GAMMA", "DYNAMIC_DQN_TARGET_FREQ", "DYNAMIC_DQN_EXPLORATION_FRACTION",
            )
        },
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
