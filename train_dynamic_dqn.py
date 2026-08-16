"""CleanRL-style DQN training for the non-steady glider environment."""

import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import config
from glider_dynamic import DynamicDiscreteActionWrapper, DynamicGliderEnv
from train_dqn import QNetwork, ReplayBuffer, linear_schedule
from train_ppo import DynamicObservationWrapper, dynamic_wind_paths


def default_model_path():
    return os.path.join(config.Q_TABLE_DIR, "dynamic_dqn_model.pth")


def default_training_log_path():
    return os.path.join(config.TRAIN_RESULT_DIR, "dynamic_dqn_training.csv")


def make_env():
    env = DynamicGliderEnv(dynamic_wind_paths(), memory_mode=True)
    env = DynamicObservationWrapper(env)
    return DynamicDiscreteActionWrapper(env)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    seed: int = config.SEED
    torch_deterministic: bool = True
    cuda: bool = True
    total_timesteps: int = config.DYNAMIC_DQN_TOTAL_TIMESTEPS
    learning_rate: float = config.DYNAMIC_DQN_LEARNING_RATE
    buffer_size: int = config.DYNAMIC_DQN_BUFFER_SIZE
    gamma: float = config.DYNAMIC_DQN_GAMMA
    tau: float = config.DYNAMIC_DQN_TAU
    target_network_frequency: int = config.DYNAMIC_DQN_TARGET_FREQ
    batch_size: int = config.DYNAMIC_DQN_BATCH_SIZE
    start_e: float = config.DYNAMIC_DQN_EPSILON_START
    end_e: float = config.DYNAMIC_DQN_EPSILON_END
    exploration_fraction: float = config.DYNAMIC_DQN_EXPLORATION_FRACTION
    learning_starts: int = config.DYNAMIC_DQN_LEARNING_STARTS
    train_frequency: int = config.DYNAMIC_DQN_TRAIN_FREQ
    model_path: str = ""
    log_path: str = ""


def train(args):
    if args.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if args.batch_size > args.buffer_size:
        raise ValueError("batch_size must not exceed buffer_size")
    if args.exploration_fraction <= 0.0:
        raise ValueError("exploration_fraction must be positive")
    if args.train_frequency <= 0 or args.target_network_frequency <= 0:
        raise ValueError("DQN frequencies must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_num_threads(config.DYNAMIC_DQN_TORCH_THREADS)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Dynamic DQN device: {device}")

    run_name = f"dynamic_dqn__{args.seed}__{int(time.time())}"
    model_path = args.model_path or default_model_path()
    log_path = args.log_path or default_training_log_path()
    writer = SummaryWriter(os.path.join(config.TRAIN_RESULT_DIR, "dynamic_dqn_runs", run_name))
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % "\n".join(
        f"|{key}|{value}|" for key, value in vars(args).items()
    ))

    env = make_env()
    q_network = QNetwork(env.observation_space.shape, env.action_space.n).to(device)
    target_network = QNetwork(env.observation_space.shape, env.action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(args.buffer_size, env.observation_space, env.action_space, device)

    observation, info = env.reset(seed=args.seed)
    episode_return = 0.0
    initial_height = info["height"]
    episode_steps = 0
    episode_rows = []
    start_time = time.time()
    
    metrics_log_interval = 1000
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            int(args.exploration_fraction * args.total_timesteps),
            global_step,
        )
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                values = q_network(torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0))
                action = int(values.argmax(dim=1).item())

        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.add(
            np.expand_dims(observation, axis=0),
            np.expand_dims(next_observation, axis=0),
            np.asarray([action]),
            np.asarray([reward]),
            np.asarray([done]),
        )
        episode_return += reward
        episode_steps += 1

        if global_step >= args.learning_starts and global_step % args.train_frequency == 0:
            data = replay_buffer.sample(args.batch_size)
            with torch.no_grad():
                target_max = target_network(data.next_observations).max(dim=1).values
                td_target = data.rewards + args.gamma * target_max * (1.0 - data.dones)
            old_values = q_network(data.observations).gather(1, data.actions).squeeze(1)
            loss = F.mse_loss(old_values, td_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % metrics_log_interval == 0:
                steps_per_second = int((global_step + 1) / (time.time() - start_time))
                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                writer.add_scalar("losses/q_values", old_values.mean().item(), global_step)
                writer.add_scalar("charts/SPS", steps_per_second, global_step)
                print(
                    f"step={global_step} td_loss={loss.item():.4f} "
                    f"q_values={old_values.mean().item():.4f} "
                    f"epsilon={epsilon:.4f} SPS={steps_per_second}"
                )

            if global_step % args.target_network_frequency == 0:
                for target_parameter, parameter in zip(target_network.parameters(), q_network.parameters()):
                    target_parameter.data.copy_(args.tau * parameter.data + (1.0 - args.tau) * target_parameter.data)

        if done:
            height_change = info["height"] - initial_height
            energy_height_change = info["energy_height"] - info["initial_energy_height"]
            episode_rows.append(
                {
                    "global_step": global_step + 1,
                    "return": episode_return,
                    "length": episode_steps,
                    "height_change": height_change,
                    "energy_height_change": energy_height_change,
                }
            )
            writer.add_scalar("charts/episodic_return", episode_return, global_step + 1)
            writer.add_scalar("charts/episodic_length", episode_steps, global_step + 1)
            writer.add_scalar("charts/height_change", height_change, global_step + 1)
            observation, info = env.reset()
            episode_return = 0.0
            initial_height = info["height"]
            episode_steps = 0
        else:
            observation = next_observation

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    torch.save(q_network.state_dict(), model_path)
    pd.DataFrame(episode_rows).to_csv(log_path, index=False)
    env.close()
    writer.close()
    print(f"Saved dynamic DQN model to {model_path}")
    return model_path


if __name__ == "__main__":
    train(tyro.cli(Args))
