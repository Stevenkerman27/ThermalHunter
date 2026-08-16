import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from glider_dynamic import DynamicGliderEnv
from train_ppo import (
    DynamicObservationWrapper,
    PPOAgent,
    default_model_path,
    dynamic_wind_paths,
)


def default_evaluation_csv_path():
    return os.path.join(config.TRAIN_RESULT_DIR, "ppo_dynamic_evaluation.csv")


def default_evaluation_plot_path():
    return os.path.join(config.TRAIN_RESULT_DIR, "ppo_dynamic_evaluation.png")


def make_scenarios(n_episodes):
    rng = np.random.default_rng(config.EVAL_SEED)
    scenarios = []
    for _ in range(n_episodes):
        x, y = rng.uniform(0.2, 0.8, size=2) * np.asarray(config.DOMAIN_SIZE[:2])
        scenarios.append(
            {
                "resettime": config.sample_start_frame(rng),
                "initial_position": np.array([x, y, rng.uniform(0.2, 0.6) * config.DOMAIN_SIZE[2]]),
                "initial_heading": rng.uniform(0.0, 2.0 * np.pi),
            }
        )
    return scenarios


def make_env():
    env = DynamicGliderEnv(dynamic_wind_paths(), memory_mode=False)
    env = DynamicObservationWrapper(env)
    return torch_action_env(env)


def torch_action_env(env):
    import gymnasium as gym

    return gym.wrappers.RescaleAction(
        env,
        min_action=np.full(2, -1.0, dtype=np.float32),
        max_action=np.full(2, 1.0, dtype=np.float32),
    )


def load_ppo_agent(model_path, device):
    agent = PPOAgent((2,), (2,)).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.eval()
    return agent


def evaluate_policy(policy_name, scenarios, agent=None, device=None):
    records = []
    random_generator = np.random.default_rng(config.EVAL_SEED + 1)
    for scenario_index, scenario in enumerate(scenarios):
        env = make_env()
        observation, info = env.reset(seed=config.EVAL_SEED + scenario_index, options=scenario)
        initial_height = info["height"]
        initial_energy_height = info["energy_height"]
        episode_return = 0.0
        step_count = 0
        terminated = truncated = False
        while not (terminated or truncated):
            if policy_name == "Random":
                action = random_generator.uniform(-1.0, 1.0, size=2).astype(np.float32)
            elif policy_name == "Cruise":
                action = np.array(
                    [
                        2.0 * config.DYNAMIC_BASELINE_SPEED_ACTION - 1.0,
                        2.0 * config.DYNAMIC_BASELINE_ROLL_ACTION - 1.0,
                    ],
                    dtype=np.float32,
                )
            else:
                with torch.no_grad():
                    observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                    action = agent.actor_mean(observation_tensor).squeeze(0).cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            step_count += 1
        records.append(
            {
                "policy": policy_name,
                "scenario": scenario_index,
                "height_change": info["height"] - initial_height,
                "energy_height_change": info["energy_height"] - initial_energy_height,
                "return": episode_return,
                "steps": step_count,
            }
        )
        env.close()
    return records


def save_plot(results, plot_path):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, column, title in (
        (axes[0], "height_change", "Height change"),
        (axes[1], "energy_height_change", "Total-energy height change"),
    ):
        results.boxplot(column=column, by="policy", ax=axis)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("m")
    figure.suptitle("")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=config.N_EVAL_EPISODES)
    parser.add_argument("--model", default=default_model_path())
    args = parser.parse_args()
    if args.n <= 0:
        raise ValueError("n must be positive")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"PPO model not found: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenarios = make_scenarios(args.n)
    agent = load_ppo_agent(args.model, device)
    records = []
    records.extend(evaluate_policy("Random", scenarios))
    records.extend(evaluate_policy("Cruise", scenarios))
    records.extend(evaluate_policy("PPO", scenarios, agent=agent, device=device))
    results = pd.DataFrame(records)
    csv_path = default_evaluation_csv_path()
    results.to_csv(csv_path, index=False)
    save_plot(results, default_evaluation_plot_path())
    print(results.groupby("policy")[["height_change", "energy_height_change", "return"]].mean().round(3))


if __name__ == "__main__":
    main()
