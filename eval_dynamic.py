import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from dynamic_observation import DynamicObservationNormalizer
from glider_discrete_simp import RBWindField
from glider_dynamic import DynamicDiscreteActionWrapper, DynamicGliderEnv
from train_dqn import QNetwork
from training_checkpoint import load_model_artifact
from train_ppo import (
    DynamicObservationWrapper,
    PPOAgent,
    default_model_path,
    dynamic_observation_shape,
    dynamic_wind_paths,
    evaluate_dynamic_policy,
    make_validation_scenarios,
)


def default_evaluation_csv_path():
    return os.path.join(config.TRAIN_RESULT_DIR, "dynamic_evaluation.csv")


def default_evaluation_plot_path():
    return os.path.join(config.TRAIN_RESULT_DIR, "dynamic_evaluation.png")


def make_scenarios(n_episodes):
    return make_validation_scenarios(n_episodes)


def make_env(wind_manager, normalizer, discrete_actions=False):
    env = DynamicGliderEnv(wind_manager=wind_manager)
    env = DynamicObservationWrapper(env, normalizer)
    if discrete_actions:
        return DynamicDiscreteActionWrapper(env)
    return torch_action_env(env)


def torch_action_env(env):
    import gymnasium as gym

    return gym.wrappers.RescaleAction(
        env,
        min_action=np.full(2, -1.0, dtype=np.float32),
        max_action=np.full(2, 1.0, dtype=np.float32),
    )


def load_ppo_agent(model_path, device):
    agent = PPOAgent(dynamic_observation_shape(), (2,)).to(device)
    state_dict, metadata = load_model_artifact(model_path, device)
    if set(metadata) != {"observation_normalizer"}:
        raise ValueError("PPO model artifact has invalid metadata")
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent, DynamicObservationNormalizer.from_dict(metadata["observation_normalizer"])


def load_dynamic_dqn_agent(model_path, device):
    action_count = config.DYNAMIC_DQN_ACTION_LEVELS ** 2
    agent = QNetwork(dynamic_observation_shape(), action_count).to(device)
    state_dict, metadata = load_model_artifact(model_path, device)
    if set(metadata) != {"observation_normalizer"}:
        raise ValueError("dynamic DQN model artifact has invalid metadata")
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent, DynamicObservationNormalizer.from_dict(metadata["observation_normalizer"])


def _evaluate_policy_single(policy_name, scenarios, wind_manager, normalizer, agent=None, device=None):
    records = []
    random_generator = np.random.default_rng(config.EVAL_SEED + 1)
    discrete_actions = policy_name in ("Random grid", "Cruise", "DQN")
    env = make_env(wind_manager, normalizer, discrete_actions=discrete_actions)
    try:
        for scenario_index, scenario in enumerate(scenarios):
            observation, info = env.reset(seed=config.EVAL_SEED + scenario_index, options=scenario)
            initial_height = info["height"]
            initial_energy_height = info["energy_height"]
            episode_return = 0.0
            step_count = 0
            terminated = truncated = False
            while not (terminated or truncated):
                if policy_name == "Random grid":
                    action = int(random_generator.integers(env.action_space.n))
                elif policy_name == "Cruise":
                    action = env.command_to_action(
                        np.array(
                            [
                                config.DYNAMIC_BASELINE_SPEED_ACTION,
                                config.DYNAMIC_BASELINE_ROLL_ACTION,
                            ],
                            dtype=np.float32,
                        )
                    )
                elif policy_name == "PPO":
                    with torch.no_grad():
                        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                        action = agent.get_deterministic_action(observation_tensor).squeeze(0).cpu().numpy()
                elif policy_name == "DQN":
                    with torch.no_grad():
                        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                        action = int(agent(observation_tensor).argmax(dim=1).item())
                else:
                    raise ValueError(f"unsupported dynamic policy: {policy_name}")
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
                    "termination_reason": info["termination_reason"],
                }
            )
    finally:
        env.close()
    return records


def evaluate_policy(policy_name, scenarios, wind_manager, normalizer, agent=None, device=None):
    discrete_actions = policy_name in ("Random grid", "Cruise", "DQN")
    random_generator = np.random.default_rng(config.EVAL_SEED + 1)
    if policy_name == "Random grid":
        def select_action(observations):
            return random_generator.integers(
                config.DYNAMIC_DQN_ACTION_LEVELS ** 2, size=len(observations)
            )
    elif policy_name == "Cruise":
        levels = config.DYNAMIC_DQN_ACTION_LEVELS
        command = np.array(
            [config.DYNAMIC_BASELINE_SPEED_ACTION, config.DYNAMIC_BASELINE_ROLL_ACTION],
            dtype=np.float32,
        )
        action = int(np.rint(command[0] * (levels - 1)) * levels + np.rint(command[1] * (levels - 1)))

        def select_action(observations):
            return np.full(len(observations), action, dtype=np.int64)
    elif policy_name == "PPO":
        def select_action(observations):
            with torch.no_grad():
                observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
                return agent.get_deterministic_action(observation_tensor).cpu().numpy()
    elif policy_name == "DQN":
        def select_action(observations):
            with torch.no_grad():
                observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
                return agent(observation_tensor).argmax(dim=1).cpu().numpy()
    else:
        raise ValueError(f"unsupported dynamic policy: {policy_name}")
    return evaluate_dynamic_policy(
        policy_name, scenarios, wind_manager, select_action, discrete_actions, normalizer
    )


def save_plot(results, plot_path):
    policies = list(results["policy"].drop_duplicates())
    colors = plt.get_cmap("tab10")(np.arange(len(policies)))
    figure, axes = plt.subplots(1, 2, figsize=(9, 6))
    for axis, column, title in (
        (axes[0], "height_change", "Height change"),
        (axes[1], "energy_height_change", "Total-energy height change"),
    ):
        values_by_policy = [
            results.loc[results["policy"] == policy, column].to_numpy()
            for policy in policies
        ]
        axis.boxplot(
            values_by_policy,
            tick_labels=policies,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "none", "edgecolor": "black"},
            medianprops={"color": "#ff7f0e", "linewidth": 2},
        )
        for position, policy, values, color in zip(
            range(1, len(policies) + 1), policies, values_by_policy, colors
        ):
            jitter = np.linspace(-0.08, 0.08, len(values))
            axis.scatter(
                position + jitter,
                values,
                s=18,
                alpha=0.55,
                color=color,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
            stats_text = (
                f"mean={np.mean(values):.1f}\n"
                f"std={np.std(values):.1f}\n"
                f"median={np.median(values):.1f}\n"
                f"n={len(values)}"
            )
            axis.text(
                position,
                1.12,
                stats_text,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=11,
                color=color,
            )
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("m")
    figure.suptitle("")
    figure.tight_layout(rect=[0, 0, 1, 0.87])
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=config.N_EVAL_EPISODES)
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--dqn-model", default=os.path.join(config.Q_TABLE_DIR, "dynamic_dqn_model.pth"))
    args = parser.parse_args()
    if args.n <= 0:
        raise ValueError("n must be positive")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"PPO model not found: {args.model}")
    if not os.path.exists(args.dqn_model):
        raise FileNotFoundError(f"dynamic DQN model not found: {args.dqn_model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenarios = make_scenarios(args.n)
    ppo_agent, ppo_normalizer = load_ppo_agent(args.model, device)
    dqn_agent, dqn_normalizer = load_dynamic_dqn_agent(args.dqn_model, device)
    wind_manager = RBWindField(dynamic_wind_paths(), memory_mode=True)
    try:
        records = []
        records.extend(evaluate_policy("Random grid", scenarios, wind_manager, ppo_normalizer))
        records.extend(evaluate_policy("Cruise", scenarios, wind_manager, ppo_normalizer))
        records.extend(evaluate_policy("PPO", scenarios, wind_manager, ppo_normalizer, agent=ppo_agent, device=device))
        records.extend(evaluate_policy("DQN", scenarios, wind_manager, dqn_normalizer, agent=dqn_agent, device=device))
    finally:
        wind_manager.close()
    results = pd.DataFrame(records)
    csv_path = default_evaluation_csv_path()
    results.to_csv(csv_path, index=False)
    save_plot(results, default_evaluation_plot_path())
    print(results.groupby("policy")[["height_change", "energy_height_change", "return"]].mean().round(3))


if __name__ == "__main__":
    main()
