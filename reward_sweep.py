import argparse
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def weight_tag(weight):
    return f"waccel_{weight:g}".replace(".", "p")


def artifact_paths(weight):
    tag = weight_tag(weight)
    return {
        "tabular_model": os.path.join(config.Q_TABLE_DIR, f"q_table_{tag}.pkl"),
        "tabular_log": os.path.join(config.TRAIN_RESULT_DIR, f"tabular_train_{tag}.csv"),
        "dqn_model": os.path.join(config.Q_TABLE_DIR, f"dqn_model_{tag}.pth"),
        "dqn_log": os.path.join(config.TRAIN_RESULT_DIR, f"dqn_train_{tag}.csv"),
        "sensor_stats": os.path.join(config.BASE_DIR, f"sensor_stats_{tag}.json"),
        "dqn_plot": os.path.join(config.TRAIN_RESULT_DIR, f"dqn_train_{tag}.png"),
        "evaluation": os.path.join(config.TRAIN_RESULT_DIR, f"evaluation_{tag}.csv"),
        "comparison_plot": os.path.join(config.TRAIN_RESULT_DIR, f"comparison_{tag}.png"),
    }


def run_command(command):
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def summarize_evaluations(weights):
    combined = None
    for weight in weights:
        paths = artifact_paths(weight)
        result = pd.read_csv(paths["evaluation"])
        if combined is None:
            combined = result[["start_frame", "Random"]].copy()
        elif not np.array_equal(combined["start_frame"].to_numpy(), result["start_frame"].to_numpy()):
            raise ValueError("evaluation scenarios differ across reward weights")
        tag = f"1:{weight:g}"
        combined[f"Tabular Q ({tag})"] = result["Tabular Q"]
        combined[f"DQN ({tag})"] = result["DQN"]

    output_csv = os.path.join(config.TRAIN_RESULT_DIR, "reward_sweep_evaluation.csv")
    combined.to_csv(output_csv, index=False)

    columns = list(combined.columns[1:])
    plot_data = [combined[column].to_numpy() for column in columns]
    fig, ax = plt.subplots(figsize=(13, 7))
    boxplot = ax.boxplot(plot_data, tick_labels=columns, showfliers=False, patch_artist=True)
    colors = ["#7f7f7f", "#1f77b4", "#d62728", "#1f77b4", "#d62728", "#1f77b4", "#d62728"]
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for index, values in enumerate(plot_data, start=1):
        ax.scatter(np.full(values.size, index), values, alpha=0.22, color=colors[index - 1], s=10)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Episodic Height Change (m)")
    ax.set_title("Reward Weight Sweep on Fixed Evaluation Scenarios")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    output_plot = os.path.join(config.TRAIN_RESULT_DIR, "reward_sweep_evaluation.png")
    fig.savefig(output_plot, dpi=300)
    plt.close(fig)

    summary = pd.DataFrame(
        {
            "policy": columns,
            "mean_height_change_m": [float(np.mean(values)) for values in plot_data],
            "std_height_change_m": [float(np.std(values)) for values in plot_data],
        }
    )
    summary_path = os.path.join(config.TRAIN_RESULT_DIR, "reward_sweep_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Sweep evaluation saved to {output_csv}")
    print(f"Sweep summary saved to {summary_path}")
    print(f"Sweep plot saved to {output_plot}")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=config.TABULAR_TOTAL_STEPS)
    parser.add_argument("--stats", type=int, default=config.SENSOR_STATS_EPISODES)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.stats <= 0:
        raise ValueError("stats must be positive")

    weights = config.REWARD_W_ACCEL_SWEEP_WEIGHTS
    for weight in weights:
        paths = artifact_paths(weight)
        print(f"\n=== Training reward ratio 1:{weight:g} ===")
        run_command(
            [
                sys.executable,
                "glider_train.py",
                "--steps",
                str(args.steps),
                "--w-accel-weight",
                str(weight),
                "--save-path",
                paths["tabular_model"],
                "--log-path",
                paths["tabular_log"],
            ]
        )
        dqn_command = [
            sys.executable,
            "train_dqn.py",
            "--total-timesteps",
            str(args.steps),
            "--sensor-stats-episodes",
            str(args.stats),
            "--reward-w-accel",
            str(weight),
            "--model-path",
            paths["dqn_model"],
            "--log-path",
            paths["dqn_log"],
            "--sensor-stats-path",
            paths["sensor_stats"],
            "--training-plot-path",
            paths["dqn_plot"],
        ]
        if args.cpu:
            dqn_command.append("--no-cuda")
        run_command(dqn_command)

    for weight in weights:
        paths = artifact_paths(weight)
        run_command(
            [
                sys.executable,
                "eval_all.py",
                "--tabular-model",
                paths["tabular_model"],
                "--dqn-model",
                paths["dqn_model"],
                "--sensor-stats",
                paths["sensor_stats"],
                "--w-accel-weight",
                str(weight),
                "--output-csv",
                paths["evaluation"],
                "--output-plot",
                paths["comparison_plot"],
            ]
        )

    summarize_evaluations(weights)


if __name__ == "__main__":
    main()
