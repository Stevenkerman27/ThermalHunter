"""Visualize dynamic glider policies without loading wind fields into memory."""

import argparse
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from glider_discrete_simp import RBWindField
from glider_dynamic import DynamicDiscreteActionWrapper, DynamicGliderEnv
from train_dqn import QNetwork
from train_ppo import (
    DynamicObservationWrapper,
    PPOAgent,
    default_model_path as default_ppo_model_path,
    dynamic_wind_paths,
)


OUTPUT_ROOT = Path(config.TRAIN_RESULT_DIR) / "dynamic_visualization"
POLICY_DIRS = {
    "PPO": "ppo",
    "DQN": "dynamic_dqn",
    "Cruise": "cruise",
    "Random grid": "random_grid",
}


def make_scenario(seed):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0.2, 0.8, size=2) * np.asarray(config.DOMAIN_SIZE[:2])
    return {
        "resettime": config.sample_start_frame(rng),
        "initial_position": np.array(
            [x, y, rng.uniform(0.2, 0.6) * config.DOMAIN_SIZE[2]],
            dtype=np.float64,
        ),
        "initial_heading": rng.uniform(0.0, 2.0 * np.pi),
    }


def load_agents(ppo_model_path, dqn_model_path, device):
    ppo_agent = PPOAgent((2,), (2,)).to(device)
    ppo_agent.load_state_dict(
        torch.load(ppo_model_path, map_location=device, weights_only=True)
    )
    ppo_agent.eval()

    dqn_action_count = config.DYNAMIC_DQN_ACTION_LEVELS ** 2
    dqn_agent = QNetwork((2,), dqn_action_count).to(device)
    dqn_agent.load_state_dict(
        torch.load(dqn_model_path, map_location=device, weights_only=True)
    )
    dqn_agent.eval()
    return ppo_agent, dqn_agent


def make_policy_env(wind_manager, policy_name):
    env = DynamicGliderEnv(wind_manager=wind_manager)
    env = DynamicObservationWrapper(env)
    if policy_name in ("DQN", "Cruise", "Random grid"):
        return DynamicDiscreteActionWrapper(env)
    return gym.wrappers.RescaleAction(
        env,
        min_action=np.full(2, -1.0, dtype=np.float32),
        max_action=np.full(2, 1.0, dtype=np.float32),
    )


def command_for_discrete_action(env, action):
    return np.asarray(env.action(action), dtype=np.float64)


def append_trajectory_row(rows, base_env, step, command, action_index, reward, done):
    info = base_env._info()
    wind = base_env._wind(base_env.position)
    rows.append(
        {
            "step": step,
            "time_s": step * config.DT_RL,
            "x": float(base_env.position[0]),
            "y": float(base_env.position[1]),
            "z": float(base_env.position[2]),
            "energy_height": info["energy_height"],
            "tas": info["tas"],
            "alpha_deg": info["alpha_deg"],
            "bank_deg": info["bank_deg"],
            "vario": info["total_energy_vario"],
            "roll_cue": info["roll_cue"],
            "wind_frame": info["wind_frame"],
            "wind_ux": float(wind[0]),
            "wind_uy": float(wind[1]),
            "wind_uz": float(wind[2]),
            "command_speed": float(command[0]),
            "command_roll": float(command[1]),
            "action_index": action_index,
            "reward": float(reward),
            "done": bool(done),
        }
    )


def run_trajectory(policy_name, scenario, wind_manager, ppo_agent, dqn_agent, device, seed, max_steps):
    env = make_policy_env(wind_manager, policy_name)
    base_env = env.unwrapped
    random_generator = np.random.default_rng(seed)
    rows = []
    try:
        observation, _ = env.reset(seed=seed, options=scenario)
        append_trajectory_row(
            rows,
            base_env,
            step=0,
            command=np.array([np.nan, np.nan]),
            action_index=None,
            reward=0.0,
            done=False,
        )

        for step in range(1, max_steps + 1):
            if policy_name == "PPO":
                with torch.no_grad():
                    observation_tensor = torch.as_tensor(
                        observation, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    raw_action = ppo_agent.actor_mean(observation_tensor).squeeze(0).cpu().numpy()
                command = np.clip((raw_action + 1.0) * 0.5, 0.0, 1.0)
                action = raw_action
                action_index = None
            elif policy_name == "DQN":
                with torch.no_grad():
                    observation_tensor = torch.as_tensor(
                        observation, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    action = int(dqn_agent(observation_tensor).argmax(dim=1).item())
                command = command_for_discrete_action(env, action)
                action_index = action
            elif policy_name == "Cruise":
                command = np.array(
                    [
                        config.DYNAMIC_BASELINE_SPEED_ACTION,
                        config.DYNAMIC_BASELINE_ROLL_ACTION,
                    ],
                    dtype=np.float64,
                )
                action = env.command_to_action(command)
                action_index = action
            else:
                action = int(random_generator.integers(env.action_space.n))
                command = command_for_discrete_action(env, action)
                action_index = action

            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            append_trajectory_row(
                rows,
                base_env,
                step=step,
                command=command,
                action_index=action_index,
                reward=reward,
                done=done,
            )
            if done:
                break
    finally:
        env.close()
    return pd.DataFrame(rows)


def unwrap_periodic(values, period):
    result = np.asarray(values, dtype=np.float64).copy()
    for index in range(1, len(result)):
        delta = result[index] - result[index - 1]
        if delta > period * 0.5:
            result[index:] -= period
        elif delta < -period * 0.5:
            result[index:] += period
    return result


def save_trajectory_plot(data, policy_name, output_path):
    x = unwrap_periodic(data["x"].to_numpy(), config.DOMAIN_SIZE[0])
    y = unwrap_periodic(data["y"].to_numpy(), config.DOMAIN_SIZE[1])
    z = data["z"].to_numpy()
    time_s = data["time_s"].to_numpy()

    figure = plt.figure(figsize=(13, 10))
    axis_3d = figure.add_subplot(221, projection="3d")
    axis_3d.plot(x, y, z, color="#2166ac", linewidth=1.8)
    axis_3d.scatter(x[0], y[0], z[0], color="#1a9850", s=45, label="Start")
    axis_3d.scatter(x[-1], y[-1], z[-1], color="#d73027", s=45, label="End")
    axis_3d.set_title(f"{policy_name} trajectory")
    axis_3d.set_xlabel("X (m)")
    axis_3d.set_ylabel("Y (m)")
    axis_3d.set_zlabel("Height (m)")
    axis_3d.legend()

    axis_energy = figure.add_subplot(222)
    axis_energy.plot(time_s, data["z"], label="Height", color="#2166ac")
    axis_energy.plot(
        time_s,
        data["energy_height"],
        label="Energy height",
        color="#e08214",
    )
    axis_energy.set_title("Height and total energy height")
    axis_energy.set_xlabel("Time (s)")
    axis_energy.set_ylabel("m")
    axis_energy.grid(alpha=0.25)
    axis_energy.legend()

    axis_controls = figure.add_subplot(223)
    axis_controls.plot(time_s, data["alpha_deg"], label="Actual AoA", color="#1b9e77")
    axis_controls.plot(time_s, data["bank_deg"], label="Actual bank", color="#d95f02")
    axis_controls.set_title("Controls")
    axis_controls.set_xlabel("Time (s)")
    axis_controls.set_ylabel("Actual angle (deg)")
    axis_controls.grid(alpha=0.25)
    axis_controls.legend(loc="upper left", fontsize=8)
    axis_commands = axis_controls.twinx()
    axis_commands.plot(
        time_s,
        data["command_speed"],
        "--",
        label="Speed command",
        color="#7570b3",
    )
    axis_commands.plot(
        time_s,
        data["command_roll"],
        "--",
        label="Roll command",
        color="#e7298a",
    )
    axis_commands.set_ylabel("Normalized command [0, 1]")
    axis_commands.set_ylim(-0.05, 1.05)
    axis_commands.legend(loc="lower right", fontsize=8)

    axis_sensors = figure.add_subplot(224)
    axis_sensors.plot(time_s, data["vario"], label="Vario", color="#4daf4a")
    axis_sensors.plot(time_s, data["roll_cue"], label="Roll cue", color="#984ea3")
    axis_sensors.plot(time_s, data["wind_uz"], label="Vertical wind", color="#ff7f00")
    axis_sensors.set_title("Dynamic observations and wind")
    axis_sensors.set_xlabel("Time (s)")
    axis_sensors.set_ylabel("Value")
    axis_sensors.grid(alpha=0.25)
    axis_sensors.legend(fontsize=8)

    figure.suptitle(f"Dynamic policy visualization: {policy_name}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_policy_map(policy_name, agent, device, output_path, grid_size=121):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    vario = np.linspace(
        -2.0 * config.DYNAMIC_VARIO_OBS_SCALE,
        2.0 * config.DYNAMIC_VARIO_OBS_SCALE,
        grid_size,
    )
    roll_cue = np.linspace(
        -2.0 * config.DYNAMIC_ROLL_CUE_OBS_SCALE,
        2.0 * config.DYNAMIC_ROLL_CUE_OBS_SCALE,
        grid_size,
    )
    vario_grid, roll_grid = np.meshgrid(vario, roll_cue)
    observations = np.stack(
        [
            vario_grid.ravel() / config.DYNAMIC_VARIO_OBS_SCALE,
            roll_grid.ravel() / config.DYNAMIC_ROLL_CUE_OBS_SCALE,
        ],
        axis=1,
    ).astype(np.float32)

    observation_tensor = torch.as_tensor(observations, device=device)
    with torch.no_grad():
        if policy_name == "PPO":
            commands = ((agent.actor_mean(observation_tensor) + 1.0) * 0.5).cpu().numpy()
            panels = [
                (commands[:, 0].reshape(vario_grid.shape), "Speed command", "viridis"),
                (commands[:, 1].reshape(vario_grid.shape), "Roll command", "coolwarm"),
            ]
        else:
            q_values = agent(observation_tensor).cpu().numpy()
            actions = q_values.argmax(axis=1)
            levels = config.DYNAMIC_DQN_ACTION_LEVELS
            speed = (actions // levels) / (levels - 1)
            roll = (actions % levels) / (levels - 1)
            panels = [
                (speed.reshape(vario_grid.shape), "Speed command", "viridis"),
                (roll.reshape(vario_grid.shape), "Roll command", "coolwarm"),
                (q_values.max(axis=1).reshape(vario_grid.shape), "Max Q value", "magma"),
            ]

    figure, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5), squeeze=False)
    for axis, (values_grid, title, cmap) in zip(axes[0], panels):
        image = axis.imshow(
            values_grid,
            origin="lower",
            aspect="auto",
            extent=[vario[0], vario[-1], roll_cue[0], roll_cue[-1]],
            cmap=cmap,
        )
        axis.set_title(title)
        axis.set_xlabel("Vario (m/s)")
        axis.set_ylabel("Roll cue (m/s)")
        figure.colorbar(image, ax=axis, shrink=0.85)
    figure.suptitle(f"Dynamic {policy_name} policy map")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_training_plot(policy_name, csv_path, output_path):
    if os.path.getsize(csv_path) == 0:
        print(f"Skipping empty training log: {csv_path}")
        return False
    try:
        data = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"Skipping empty training log: {csv_path}")
        return False
    if data.empty:
        print(f"Skipping empty training log: {csv_path}")
        return False
    required = ["return", "height_change", "energy_height_change"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"training log missing columns: {missing}")

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis, column, title in zip(
        axes,
        required,
        ["Episode return", "Height change", "Energy height change"],
    ):
        values = data[column].to_numpy(dtype=float)
        axes_index = np.arange(len(values))
        axis.plot(axes_index, values, alpha=0.25, color="#2166ac")
        if len(values) >= 10:
            window = min(50, len(values))
            moving_average = pd.Series(values).rolling(window).mean()
            axis.plot(moving_average, color="#b2182b", linewidth=2, label=f"Moving average ({window})")
            axis.legend()
        axis.set_ylabel(column)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Episode")
    figure.suptitle(f"Dynamic {policy_name} training")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--ppo-model", default=default_ppo_model_path())
    parser.add_argument("--dqn-model", default=os.path.join(config.Q_TABLE_DIR, "dynamic_dqn_model.pth"))
    parser.add_argument("--skip-maps", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    if args.n <= 0 or args.max_steps <= 0:
        raise ValueError("n and max-steps must be positive")
    if not os.path.exists(args.ppo_model):
        raise FileNotFoundError(f"PPO model not found: {args.ppo_model}")
    if not os.path.exists(args.dqn_model):
        raise FileNotFoundError(f"dynamic DQN model not found: {args.dqn_model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_agent, dqn_agent = load_agents(args.ppo_model, args.dqn_model, device)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.skip_maps:
        save_policy_map("PPO", ppo_agent, device, OUTPUT_ROOT / POLICY_DIRS["PPO"] / "policy_map.png")
        save_policy_map("DQN", dqn_agent, device, OUTPUT_ROOT / POLICY_DIRS["DQN"] / "policy_map.png")

    if not args.skip_training:
        training_logs = {
            "PPO": os.path.join(config.TRAIN_RESULT_DIR, "ppo_dynamic_training.csv"),
            "DQN": os.path.join(config.TRAIN_RESULT_DIR, "dynamic_dqn_training.csv"),
        }
        for policy_name, csv_path in training_logs.items():
            if not os.path.exists(csv_path):
                print(f"Skipping missing training log: {csv_path}")
                continue
            output_path = OUTPUT_ROOT / POLICY_DIRS[policy_name] / "training.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_training_plot(policy_name, csv_path, output_path)

    policies = ("PPO", "DQN", "Cruise", "Random grid")
    wind_manager = RBWindField(dynamic_wind_paths(), memory_mode=False)
    try:
        for policy_index, policy_name in enumerate(policies):
            output_dir = OUTPUT_ROOT / POLICY_DIRS[policy_name]
            output_dir.mkdir(parents=True, exist_ok=True)
            summaries = []
            for scenario_index in range(args.n):
                scenario_seed = config.EVAL_SEED + scenario_index
                data = run_trajectory(
                    policy_name,
                    make_scenario(scenario_seed),
                    wind_manager,
                    ppo_agent,
                    dqn_agent,
                    device,
                    seed=scenario_seed + policy_index,
                    max_steps=args.max_steps,
                )
                suffix = "" if args.n == 1 else f"_{scenario_index + 1:03d}"
                data.to_csv(output_dir / f"trajectory{suffix}.csv", index=False)
                save_trajectory_plot(data, policy_name, output_dir / f"trajectory{suffix}.png")
                summaries.append(
                    {
                        "scenario": scenario_index,
                        "steps": len(data) - 1,
                        "height_change": data["z"].iloc[-1] - data["z"].iloc[0],
                        "energy_height_change": data["energy_height"].iloc[-1] - data["energy_height"].iloc[0],
                        "return": data["reward"].sum(),
                    }
                )
                print(
                    f"{policy_name} scenario={scenario_index}: steps={len(data) - 1}, "
                    f"height_change={data['z'].iloc[-1] - data['z'].iloc[0]:.2f} m, "
                    f"output={output_dir}"
                )
            pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)
    finally:
        wind_manager.close()


if __name__ == "__main__":
    main()
