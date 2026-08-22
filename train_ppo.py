"""CleanRL-style PPO training for the non-steady glider environment."""

import glob
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import config
from dynamic_observation import (
    DynamicObservationNormalizer,
    default_dynamic_observation_stats_path,
    load_dynamic_observation_normalizer,
    save_dynamic_observation_normalizer,
)
from training_checkpoint import save_model_artifact, save_training_checkpoint
from glider_dynamic import (
    DynamicDiscreteActionWrapper,
    DynamicGliderBatchEnv,
    DynamicGliderEnv,
    dynamic_discrete_action_commands,
)


def default_model_path():
    return os.path.join(config.Q_TABLE_DIR, "ppo_dynamic_model.pth")


def default_training_log_path():
    return os.path.join(config.TRAIN_RESULT_DIR, "ppo_dynamic_training.csv")


def default_update_log_path():
    return os.path.join(config.TRAIN_RESULT_DIR, "ppo_dynamic_updates.csv")


def normalized_dynamic_observation(observation, normalizer):
    return normalizer.normalize(observation)


def dynamic_observation_shape():
    return (4,)


class DynamicObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, normalizer):
        super().__init__(env)
        self.normalizer = normalizer
        self.observation_space = gym.spaces.Box(
            low=np.full(dynamic_observation_shape(), -np.inf, dtype=np.float32),
            high=np.full(dynamic_observation_shape(), np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation):
        return normalized_dynamic_observation(observation, self.normalizer)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        obs_dim = int(np.prod(observation_shape))
        action_dim = int(np.prod(action_shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, observation):
        return self.critic(observation)

    def get_action_and_value(self, observation, action=None):
        action_mean = self.actor_mean(observation)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        distribution = Normal(action_mean, torch.exp(action_logstd))
        if action is None:
            action = torch.tanh(distribution.sample())
        action = torch.clamp(action, -1.0 + config.PPO_SQUASH_EPSILON, 1.0 - config.PPO_SQUASH_EPSILON)
        latent_action = torch.atanh(action)
        logprob = distribution.log_prob(latent_action) - torch.log(1.0 - action.square() + config.PPO_SQUASH_EPSILON)
        return (
            action,
            logprob.sum(1),
            distribution.entropy().sum(1),
            self.critic(observation),
        )

    def get_deterministic_action(self, observation):
        return torch.tanh(self.actor_mean(observation))


def dynamic_wind_paths():
    paths = sorted(glob.glob(os.path.join(config.WIND_DIR, "snapshots_s*.h5")), key=config.natural_key)
    if not paths:
        raise FileNotFoundError("no wind snapshots found")
    return paths


def make_validation_scenarios(n_episodes, seed=None):
    scenario_seed = config.EVAL_SEED if seed is None else seed
    rng = np.random.default_rng(scenario_seed)
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


def collect_dynamic_observation_normalizer(h5_paths, seed):
    if config.DYNAMIC_NORMALIZATION_STEPS <= 0:
        raise ValueError("config.DYNAMIC_NORMALIZATION_STEPS must be positive")
    env = DynamicGliderBatchEnv(
        config.DYNAMIC_NUM_ENVS,
        h5_paths,
        memory_mode=True,
        autoreset=True,
    )
    rng = np.random.default_rng(seed)
    observations = []
    try:
        raw_observations, _ = env.reset(seed=seed)
        observations.append(raw_observations)
        for _ in range(config.DYNAMIC_NORMALIZATION_STEPS):
            commands = rng.uniform(0.0, 1.0, size=(env.num_envs, 2))
            raw_observations, _, _, _, _ = env.step(commands)
            observations.append(raw_observations)
    finally:
        env.close()
    return DynamicObservationNormalizer.from_samples(np.concatenate(observations, axis=0))


def load_or_collect_dynamic_observation_normalizer(h5_paths, seed):
    stats_path = default_dynamic_observation_stats_path()
    if os.path.isfile(stats_path):
        print(f"Loaded dynamic observation normalization from {stats_path}")
        return load_dynamic_observation_normalizer(stats_path)
    normalizer = collect_dynamic_observation_normalizer(h5_paths, seed)
    save_dynamic_observation_normalizer(normalizer, stats_path)
    print(f"Saved dynamic observation normalization to {stats_path}")
    return normalizer


def annealed_entropy_coefficient(initial_coefficient, iteration, num_iterations):
    if num_iterations <= 0 or not 1 <= iteration <= num_iterations:
        raise ValueError("entropy schedule iteration must be within a positive iteration count")
    if num_iterations == 1:
        return initial_coefficient
    fraction = 1.0 - (iteration - 1.0) / (num_iterations - 1.0)
    return config.PPO_ENT_COEF_FINAL + (
        initial_coefficient - config.PPO_ENT_COEF_FINAL
    ) * fraction


def evaluate_dynamic_policy(
    policy_name,
    scenarios,
    wind_manager,
    action_selector,
    discrete_actions,
    normalizer,
    scenario_seed=None,
):
    reset_seed = config.EVAL_SEED if scenario_seed is None else scenario_seed
    records = []
    for start in range(0, len(scenarios), config.DYNAMIC_NUM_ENVS):
        batch_scenarios = scenarios[start:start + config.DYNAMIC_NUM_ENVS]
        env = DynamicGliderBatchEnv(
            len(batch_scenarios), wind_manager=wind_manager, autoreset=False
        )
        try:
            raw_observations, _ = env.reset(
                seed=reset_seed + start,
                options=batch_scenarios,
            )
            observations = normalized_dynamic_observation(raw_observations, normalizer)
            episode_returns = np.zeros(env.num_envs, dtype=np.float64)
            episode_steps = np.zeros(env.num_envs, dtype=np.int64)
            active = np.ones(env.num_envs, dtype=bool)
            while np.any(active):
                selected_actions = np.asarray(action_selector(observations))
                if discrete_actions:
                    if selected_actions.shape != (env.num_envs,):
                        raise ValueError("batched DQN evaluator must return shape (num_envs,)")
                    commands = dynamic_discrete_action_commands(selected_actions)
                else:
                    if selected_actions.shape != (env.num_envs, 2):
                        raise ValueError("batched PPO evaluator must return shape (num_envs, 2)")
                    commands = np.clip((selected_actions + 1.0) * 0.5, 0.0, 1.0)
                raw_observations, rewards, terminated, truncated, infos = env.step(
                    commands, active_mask=active
                )
                episode_returns[active] += rewards[active]
                episode_steps[active] += 1
                finished = active & np.logical_or(terminated, truncated)
                for env_index in np.flatnonzero(finished):
                    records.append(
                        {
                            "policy": policy_name,
                            "scenario": start + env_index,
                            "return": episode_returns[env_index],
                            "height_change": infos["height"][env_index] - infos["initial_height"][env_index],
                            "energy_height_change": (
                                infos["energy_height"][env_index]
                                - infos["initial_energy_height"][env_index]
                            ),
                            "steps": episode_steps[env_index],
                            "termination_reason": infos["termination_reason"][env_index],
                        }
                    )
                active &= ~finished
                observations = normalized_dynamic_observation(raw_observations, normalizer)
        finally:
            env.close()
    return records


def summarize_validation(global_step, records):
    results = pd.DataFrame(records)
    energy = results["energy_height_change"]
    summary = {
        "global_step": global_step,
        "mean_return": results["return"].mean(),
        "mean_energy_height_change": energy.mean(),
        "median_energy_height_change": energy.median(),
        "energy_height_change_stderr": energy.sem(),
        "mean_height_change": results["height_change"].mean(),
        "mean_steps": results["steps"].mean(),
    }
    for reason in ("altitude_low", "altitude_high", "wind_end", "numerical_divergence"):
        summary[f"termination_{reason}_fraction"] = (results["termination_reason"] == reason).mean()
    return summary


def make_env(seed, capture_video, run_name, normalizer):
    def thunk():
        env = DynamicGliderEnv(dynamic_wind_paths(), memory_mode=True)
        env = DynamicObservationWrapper(env, normalizer)
        env = gym.wrappers.RescaleAction(
            env,
            min_action=np.full(2, -1.0, dtype=np.float32),
            max_action=np.full(2, 1.0, dtype=np.float32),
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            env = gym.wrappers.RecordVideo(env, os.path.join(config.TRAIN_RESULT_DIR, "ppo_videos", run_name))
        env.action_space.seed(seed)
        return env

    return thunk


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    seed: int = config.SEED
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    total_timesteps: int = config.PPO_TOTAL_TIMESTEPS
    learning_rate: float = config.PPO_LEARNING_RATE
    num_envs: int = config.DYNAMIC_NUM_ENVS
    num_steps: int = config.PPO_NUM_STEPS
    anneal_lr: bool = True
    gamma: float = config.PPO_GAMMA
    gae_lambda: float = config.PPO_GAE_LAMBDA
    num_minibatches: int = config.PPO_NUM_MINIBATCHES
    update_epochs: int = config.PPO_UPDATE_EPOCHS
    norm_adv: bool = True
    clip_coef: float = config.PPO_CLIP_COEF
    clip_vloss: bool = config.PPO_CLIP_VLOSS
    ent_coef: float = config.PPO_ENT_COEF
    vf_coef: float = config.PPO_VF_COEF
    max_grad_norm: float = config.PPO_MAX_GRAD_NORM
    target_kl: float | None = None
    model_path: str = ""
    log_path: str = ""
    update_log_path: str = ""


def train(args):
    if args.num_envs != config.DYNAMIC_NUM_ENVS:
        raise ValueError("dynamic PPO environment count is defined by config.DYNAMIC_NUM_ENVS")
    if args.capture_video:
        raise ValueError("video capture is not supported by the batched dynamic environment")
    if args.total_timesteps < args.num_envs * args.num_steps:
        raise ValueError("total_timesteps must cover one full batched PPO rollout")
    if (args.num_envs * args.num_steps) % args.num_minibatches != 0:
        raise ValueError("the total PPO rollout batch must be divisible by num_minibatches")
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    run_name = f"dynamic_ppo__{args.seed}__{int(time.time())}"
    model_path = args.model_path or default_model_path()
    log_path = args.log_path or default_training_log_path()
    update_log_path = args.update_log_path or default_update_log_path()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_num_threads(config.PPO_TORCH_THREADS)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"PPO device: {device}")

    writer = SummaryWriter(os.path.join(config.TRAIN_RESULT_DIR, "ppo_runs", run_name))
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % "\n".join(
        f"|{key}|{value}|" for key, value in vars(args).items()
    ))
    h5_paths = dynamic_wind_paths()
    normalizer = load_or_collect_dynamic_observation_normalizer(h5_paths, args.seed)
    model_metadata = {"observation_normalizer": normalizer.to_dict()}
    writer.add_text("observation_normalizer", str(normalizer.to_dict()))
    envs = DynamicGliderBatchEnv(
        args.num_envs,
        h5_paths,
        memory_mode=True,
        autoreset=True,
    )
    agent = PPOAgent(envs.single_observation_space.shape, envs.single_action_space.shape).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    observations = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_observation, _ = envs.reset(seed=args.seed)
    next_observation = torch.as_tensor(
        normalized_dynamic_observation(next_observation, normalizer), dtype=torch.float32, device=device
    )
    next_done = torch.zeros(args.num_envs, device=device)
    global_step = 0
    start_time = time.time()
    episode_rows = []
    update_rows = []
    next_checkpoint_step = config.DYNAMIC_CHECKPOINT_INTERVAL
    episode_actions = [[] for _ in range(args.num_envs)]
    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
    next_report_episode = config.DYNAMIC_REPORT_EPISODES

    for iteration in range(1, num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
        entropy_coef = annealed_entropy_coefficient(args.ent_coef, iteration, num_iterations)

        for step in range(args.num_steps):
            global_step += args.num_envs
            observations[step] = next_observation
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_observation)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            actions_np = action.cpu().numpy()
            commands = np.clip((actions_np + 1.0) * 0.5, 0.0, 1.0)
            next_observation_np, reward, terminations, truncations, infos = envs.step(commands)
            for env_index, command in enumerate(commands):
                episode_actions[env_index].append(command)
            episode_returns += reward
            episode_lengths += 1
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_observation = torch.as_tensor(
                normalized_dynamic_observation(next_observation_np, normalizer), dtype=torch.float32, device=device
            )
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

            if np.any(next_done_np):
                for env_index in np.flatnonzero(next_done_np):
                    action_values = np.asarray(episode_actions[env_index], dtype=np.float64)
                    episode_rows.append(
                        {
                            "global_step": global_step,
                            "return": float(episode_returns[env_index]),
                            "length": int(episode_lengths[env_index]),
                            "height_change": float(infos["height"][env_index] - infos["initial_height"][env_index]),
                            "energy_height_change": float(
                                infos["energy_height"][env_index]
                                - infos["initial_energy_height"][env_index]
                            ),
                            "termination_reason": infos["termination_reason"][env_index],
                            "speed_action_mean": float(action_values[:, 0].mean()),
                            "speed_action_std": float(action_values[:, 0].std()),
                            "roll_action_mean": float(action_values[:, 1].mean()),
                            "roll_action_std": float(action_values[:, 1].std()),
                            "action_saturation_fraction": float(
                                (np.abs(action_values) >= 1.0 - config.DYNAMIC_ACTION_SATURATION_MARGIN).mean()
                            ),
                        }
                    )
                    episode_actions[env_index] = []
                    episode_returns[env_index] = 0.0
                    episode_lengths[env_index] = 0
                    writer.add_scalar(
                        "charts/episodic_return",
                        float(episode_rows[-1]["return"]),
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_length",
                        int(episode_rows[-1]["length"]),
                        global_step,
                    )

        with torch.no_grad():
            next_value = agent.get_value(next_observation).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            last_gae_lambda = 0.0
            for step in reversed(range(args.num_steps)):
                if step == args.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones[step + 1]
                    next_values = values[step + 1]
                delta = rewards[step] + args.gamma * next_values * next_nonterminal - values[step]
                advantages[step] = last_gae_lambda = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae_lambda
            returns = advantages + values

        batch_observations = observations.reshape((-1,) + envs.single_observation_space.shape)
        batch_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        batch_logprobs = logprobs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)
        batch_indices = np.arange(batch_size)
        clip_fractions = []
        approx_kls = []

        for _ in range(args.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, batch_size, minibatch_size):
                indices = batch_indices[start:start + minibatch_size]
                _, new_logprob, entropy, new_value = agent.get_action_and_value(batch_observations[indices], batch_actions[indices])
                log_ratio = new_logprob - batch_logprobs[indices]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    clip_fractions.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                    approx_kls.append(((ratio - 1.0) - log_ratio).mean().item())

                minibatch_advantages = batch_advantages[indices]
                if args.norm_adv:
                    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)
                policy_loss_1 = -minibatch_advantages * ratio
                policy_loss_2 = -minibatch_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                policy_loss = torch.maximum(policy_loss_1, policy_loss_2).mean()

                new_value = new_value.view(-1)
                if args.clip_vloss:
                    unclipped_value_loss = (new_value - batch_returns[indices]) ** 2
                    clipped_value = batch_values[indices] + torch.clamp(new_value - batch_values[indices], -args.clip_coef, args.clip_coef)
                    value_loss = 0.5 * torch.maximum(unclipped_value_loss, (clipped_value - batch_returns[indices]) ** 2).mean()
                else:
                    value_loss = 0.5 * ((new_value - batch_returns[indices]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss - entropy_coef * entropy_loss + args.vf_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        explained_variance = 1.0 - np.var((batch_returns - batch_values).detach().cpu().numpy()) / np.var(batch_returns.detach().cpu().numpy())
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/entropy_coef", entropy_coef, global_step)
        writer.add_scalar("losses/action_std", torch.exp(agent.actor_logstd).mean().item(), global_step)
        writer.add_scalar("losses/clip_fraction", float(np.mean(clip_fractions)), global_step)
        writer.add_scalar("losses/approx_kl", float(np.mean(approx_kls)), global_step)
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        update_rows.append(
            {
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy_loss.item(),
                "entropy_coef": entropy_coef,
                "action_std": torch.exp(agent.actor_logstd).mean().item(),
                "approx_kl": float(np.mean(approx_kls)),
                "clip_fraction": float(np.mean(clip_fractions)),
                "explained_variance": explained_variance,
            }
        )
        if global_step >= next_checkpoint_step:
            checkpoint_path = save_training_checkpoint(
                agent,
                model_path,
                global_step,
                (
                    (episode_rows, log_path),
                    (update_rows, update_log_path),
                ),
                model_metadata,
            )
            print(f"Saved PPO checkpoint to {checkpoint_path}")
            while next_checkpoint_step <= global_step:
                next_checkpoint_step += config.DYNAMIC_CHECKPOINT_INTERVAL
        while len(episode_rows) >= next_report_episode:
            print(
                f"step={global_step} policy_loss={policy_loss.item():.4f} "
                f"value_loss={value_loss.item():.4f} episodes={next_report_episode}"
            )
            next_report_episode += config.DYNAMIC_REPORT_EPISODES

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    save_model_artifact(agent, model_path, model_metadata)
    pd.DataFrame(episode_rows).to_csv(log_path, index=False)
    pd.DataFrame(update_rows).to_csv(update_log_path, index=False)
    envs.close()
    writer.close()
    print(f"Saved PPO model to {model_path}")
    return model_path


if __name__ == "__main__":
    train(tyro.cli(Args))
