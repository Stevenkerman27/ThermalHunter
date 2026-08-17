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
from glider_dynamic import DynamicGliderEnv


def default_model_path():
    return os.path.join(config.Q_TABLE_DIR, "ppo_dynamic_model.pth")


def default_training_log_path():
    return os.path.join(config.TRAIN_RESULT_DIR, "ppo_dynamic_training.csv")


def normalized_dynamic_observation(observation):
    scale = np.array(
        [config.DYNAMIC_VARIO_OBS_SCALE, config.DYNAMIC_ROLL_CUE_OBS_SCALE],
        dtype=np.float32,
    )
    return np.asarray(observation, dtype=np.float32) / scale


class DynamicObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.full(2, -np.inf, dtype=np.float32),
            high=np.full(2, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation):
        return normalized_dynamic_observation(observation)


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
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, observation):
        return self.critic(observation)

    def get_action_and_value(self, observation, action=None):
        action_mean = self.actor_mean(observation)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        distribution = Normal(action_mean, torch.exp(action_logstd))
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action).sum(1),
            distribution.entropy().sum(1),
            self.critic(observation),
        )


def dynamic_wind_paths():
    paths = sorted(glob.glob(os.path.join(config.WIND_DIR, "snapshots_s*.h5")), key=config.natural_key)
    if not paths:
        raise FileNotFoundError("no wind snapshots found")
    return paths


def make_env(seed, capture_video, run_name):
    def thunk():
        env = DynamicGliderEnv(dynamic_wind_paths(), memory_mode=True)
        env = DynamicObservationWrapper(env)
        env = gym.wrappers.RescaleAction(
            env,
            min_action=np.full(2, -1.0, dtype=np.float32),
            max_action=np.full(2, 1.0, dtype=np.float32),
        )
        env = gym.wrappers.ClipAction(env)
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
    num_envs: int = 1
    num_steps: int = config.PPO_NUM_STEPS
    anneal_lr: bool = True
    gamma: float = config.PPO_GAMMA
    gae_lambda: float = config.PPO_GAE_LAMBDA
    num_minibatches: int = config.PPO_NUM_MINIBATCHES
    update_epochs: int = config.PPO_UPDATE_EPOCHS
    norm_adv: bool = True
    clip_coef: float = config.PPO_CLIP_COEF
    clip_vloss: bool = True
    ent_coef: float = config.PPO_ENT_COEF
    vf_coef: float = config.PPO_VF_COEF
    max_grad_norm: float = config.PPO_MAX_GRAD_NORM
    log_interval: int = config.PPO_LOG_INTERVAL
    target_kl: float | None = None
    model_path: str = ""
    log_path: str = ""


def train(args):
    if args.num_envs != 1:
        raise ValueError("dynamic PPO supports exactly one environment to avoid duplicate wind-field memory")
    if args.total_timesteps < args.num_steps:
        raise ValueError("total_timesteps must be at least num_steps")
    if args.num_steps % args.num_minibatches != 0:
        raise ValueError("num_steps must be divisible by num_minibatches")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive")

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    run_name = f"dynamic_ppo__{args.seed}__{int(time.time())}"
    model_path = args.model_path or default_model_path()
    log_path = args.log_path or default_training_log_path()

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
    envs = gym.vector.SyncVectorEnv([make_env(args.seed, args.capture_video, run_name)])
    agent = PPOAgent(envs.single_observation_space.shape, envs.single_action_space.shape).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    observations = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_observation, _ = envs.reset(seed=args.seed)
    next_observation = torch.as_tensor(next_observation, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    global_step = 0
    start_time = time.time()
    episode_rows = []

    for iteration in range(1, num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = fraction * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            observations[step] = next_observation
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_observation)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_observation_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_observation = torch.as_tensor(next_observation_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

            if "episode" in infos:
                episode = infos["episode"]
                for env_index in np.flatnonzero(infos["_episode"]):
                    episode_rows.append(
                        {
                            "global_step": global_step,
                            "return": float(episode["r"][env_index]),
                            "length": int(episode["l"][env_index]),
                            "height_change": float(infos["height"][env_index] - infos["initial_height"][env_index]),
                            "energy_height_change": float(
                                infos["energy_height"][env_index]
                                - infos["initial_energy_height"][env_index]
                            ),
                        }
                    )
                    writer.add_scalar(
                        "charts/episodic_return",
                        float(episode["r"][env_index]),
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_length",
                        int(episode["l"][env_index]),
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

        for _ in range(args.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, batch_size, minibatch_size):
                indices = batch_indices[start:start + minibatch_size]
                _, new_logprob, entropy, new_value = agent.get_action_and_value(batch_observations[indices], batch_actions[indices])
                log_ratio = new_logprob - batch_logprobs[indices]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    clip_fractions.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

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
                loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        explained_variance = 1.0 - np.var((batch_returns - batch_values).detach().cpu().numpy()) / np.var(batch_returns.detach().cpu().numpy())
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/clip_fraction", float(np.mean(clip_fractions)), global_step)
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if iteration % args.log_interval == 0 or iteration == num_iterations:
            print(
                f"step={global_step} policy_loss={policy_loss.item():.4f} "
                f"value_loss={value_loss.item():.4f} episodes={len(episode_rows)}"
            )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    torch.save(agent.state_dict(), model_path)
    pd.DataFrame(episode_rows).to_csv(log_path, index=False)
    envs.close()
    writer.close()
    print(f"Saved PPO model to {model_path}")
    return model_path


if __name__ == "__main__":
    train(tyro.cli(Args))
