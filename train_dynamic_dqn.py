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
from training_checkpoint import save_model_artifact, save_training_checkpoint
from glider_dynamic import (
    DynamicDiscreteActionBatchWrapper,
    DynamicDiscreteActionWrapper,
    DynamicGliderBatchEnv,
    DynamicGliderEnv,
    dynamic_discrete_action_commands,
)
from train_dqn import QNetwork, ReplayBuffer, linear_schedule
from train_ppo import (
    DynamicObservationWrapper,
    dynamic_wind_paths,
    load_or_collect_dynamic_observation_normalizer,
    normalized_dynamic_observation,
)


def default_model_path():
    return os.path.join(config.Q_TABLE_DIR, "dynamic_dqn_model.pth")


def default_training_log_path():
    return os.path.join(config.TRAIN_RESULT_DIR, "dynamic_dqn_training.csv")


def default_update_log_path():
    return os.path.join(config.TRAIN_RESULT_DIR, "dynamic_dqn_updates.csv")


def make_env(normalizer):
    env = DynamicGliderEnv(dynamic_wind_paths(), memory_mode=True)
    env = DynamicObservationWrapper(env, normalizer)
    return DynamicDiscreteActionWrapper(env)


def make_batch_env(h5_paths):
    return DynamicDiscreteActionBatchWrapper(
        DynamicGliderBatchEnv(
            config.DYNAMIC_NUM_ENVS,
            h5_paths,
            memory_mode=True,
            autoreset=True,
        )
    )


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
    num_envs: int = config.DYNAMIC_NUM_ENVS
    model_path: str = ""
    log_path: str = ""
    update_log_path: str = ""


def _train_single(args):
    raise RuntimeError("single-environment dynamic DQN training has been removed")

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
    update_log_path = args.update_log_path or default_update_log_path()
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
    update_rows = []
    next_checkpoint_step = config.DYNAMIC_CHECKPOINT_INTERVAL
    episode_actions = []
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
        episode_actions.append(env.action(action))
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
            update_rows.append(
                {
                    "global_step": global_step + 1,
                    "td_loss": loss.item(),
                    "mean_q_value": old_values.mean().item(),
                    "epsilon": epsilon,
                    "steps_per_second": int((global_step + 1) / (time.time() - start_time)),
                }
            )

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
                    "termination_reason": info["termination_reason"],
                    "speed_action_mean": float(np.mean(episode_actions, axis=0)[0]),
                    "speed_action_std": float(np.std(episode_actions, axis=0)[0]),
                    "roll_action_mean": float(np.mean(episode_actions, axis=0)[1]),
                    "roll_action_std": float(np.std(episode_actions, axis=0)[1]),
                    "action_saturation_fraction": float(
                        (np.logical_or(
                            np.asarray(episode_actions) <= config.DYNAMIC_ACTION_SATURATION_MARGIN,
                            np.asarray(episode_actions) >= 1.0 - config.DYNAMIC_ACTION_SATURATION_MARGIN,
                        )).mean()
                    ),
                }
            )
            writer.add_scalar("charts/episodic_return", episode_return, global_step + 1)
            writer.add_scalar("charts/episodic_length", episode_steps, global_step + 1)
            writer.add_scalar("charts/height_change", height_change, global_step + 1)
            observation, info = env.reset()
            episode_return = 0.0
            initial_height = info["height"]
            episode_steps = 0
            episode_actions = []
        else:
            observation = next_observation

        completed_steps = global_step + 1
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    torch.save(q_network.state_dict(), model_path)
    pd.DataFrame(episode_rows).to_csv(log_path, index=False)
    pd.DataFrame(update_rows).to_csv(update_log_path, index=False)
    env.close()
    writer.close()
    print(f"Saved dynamic DQN model to {model_path}")
    return model_path


def train(args):
    if args.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if args.num_envs != config.DYNAMIC_NUM_ENVS:
        raise ValueError("dynamic DQN environment count is defined by config.DYNAMIC_NUM_ENVS")
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
    update_log_path = args.update_log_path or default_update_log_path()
    writer = SummaryWriter(os.path.join(config.TRAIN_RESULT_DIR, "dynamic_dqn_runs", run_name))
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % "\n".join(
        f"|{key}|{value}|" for key, value in vars(args).items()
    ))

    h5_paths = dynamic_wind_paths()
    normalizer = load_or_collect_dynamic_observation_normalizer(h5_paths, args.seed)
    model_metadata = {"observation_normalizer": normalizer.to_dict()}
    writer.add_text("observation_normalizer", str(normalizer.to_dict()))
    env = make_batch_env(h5_paths)
    q_network = QNetwork(env.single_observation_space.shape, env.single_action_space.n).to(device)
    target_network = QNetwork(env.single_observation_space.shape, env.single_action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(
        args.buffer_size, env.single_observation_space, env.single_action_space, device
    )

    observation, _ = env.reset(seed=args.seed)
    observation = normalized_dynamic_observation(observation, normalizer)
    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
    episode_actions = [[] for _ in range(args.num_envs)]
    episode_rows = []
    update_rows = []
    next_checkpoint_step = config.DYNAMIC_CHECKPOINT_INTERVAL
    next_report_episode = config.DYNAMIC_REPORT_EPISODES
    global_step = 0
    next_train_step = args.learning_starts
    next_target_step = args.target_network_frequency
    start_time = time.time()
    last_loss = None
    last_q_value = None

    try:
        while global_step < args.total_timesteps:
            active_count = min(args.num_envs, args.total_timesteps - global_step)
            active_mask = np.arange(args.num_envs) < active_count
            step_indices = global_step + np.arange(args.num_envs)
            epsilon_values = np.maximum(
                (args.end_e - args.start_e)
                / int(args.exploration_fraction * args.total_timesteps)
                * step_indices
                + args.start_e,
                args.end_e,
            )
            with torch.no_grad():
                observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
                actions = q_network(observation_tensor).argmax(dim=1).cpu().numpy()
            exploratory = np.random.random(args.num_envs) < epsilon_values
            actions[exploratory] = np.random.randint(
                env.single_action_space.n, size=int(exploratory.sum())
            )
            commands = dynamic_discrete_action_commands(actions)
            next_observation_raw, rewards, terminations, truncations, infos = env.step(
                actions, active_mask=active_mask
            )
            done = np.logical_or(terminations, truncations)
            replay_buffer.add(
                observation[active_mask],
                normalized_dynamic_observation(next_observation_raw, normalizer)[active_mask],
                actions[active_mask],
                rewards[active_mask],
                done[active_mask],
            )
            for env_index in np.flatnonzero(active_mask):
                episode_actions[env_index].append(commands[env_index])
            episode_returns[active_mask] += rewards[active_mask]
            episode_lengths[active_mask] += 1
            for env_index in np.flatnonzero(done & active_mask):
                action_values = np.asarray(episode_actions[env_index], dtype=np.float64)
                episode_rows.append(
                    {
                        "global_step": global_step + active_count,
                        "return": float(episode_returns[env_index]),
                        "length": int(episode_lengths[env_index]),
                        "height_change": float(infos["height"][env_index] - infos["initial_height"][env_index]),
                        "energy_height_change": float(
                            infos["energy_height"][env_index] - infos["initial_energy_height"][env_index]
                        ),
                        "termination_reason": infos["termination_reason"][env_index],
                        "speed_action_mean": float(action_values[:, 0].mean()),
                        "speed_action_std": float(action_values[:, 0].std()),
                        "roll_action_mean": float(action_values[:, 1].mean()),
                        "roll_action_std": float(action_values[:, 1].std()),
                        "action_saturation_fraction": float(
                            np.logical_or(
                                action_values <= config.DYNAMIC_ACTION_SATURATION_MARGIN,
                                action_values >= 1.0 - config.DYNAMIC_ACTION_SATURATION_MARGIN,
                            ).mean()
                        ),
                    }
                )
                writer.add_scalar("charts/episodic_return", episode_returns[env_index], global_step + active_count)
                writer.add_scalar("charts/episodic_length", episode_lengths[env_index], global_step + active_count)
                writer.add_scalar(
                    "charts/height_change",
                    episode_rows[-1]["height_change"],
                    global_step + active_count,
                )
                episode_returns[env_index] = 0.0
                episode_lengths[env_index] = 0
                episode_actions[env_index] = []

            observation = normalized_dynamic_observation(next_observation_raw, normalizer)
            global_step += active_count
            while next_train_step <= global_step:
                data = replay_buffer.sample(args.batch_size)
                with torch.no_grad():
                    target_max = target_network(data.next_observations).max(dim=1).values
                    td_target = data.rewards + args.gamma * target_max * (1.0 - data.dones)
                old_values = q_network(data.observations).gather(1, data.actions).squeeze(1)
                loss = F.mse_loss(old_values, td_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = loss.item()
                last_q_value = old_values.mean().item()
                update_rows.append(
                    {
                        "global_step": next_train_step,
                        "td_loss": last_loss,
                        "mean_q_value": last_q_value,
                        "epsilon": float(epsilon_values[0]),
                        "steps_per_second": int(global_step / (time.time() - start_time)),
                    }
                )
                next_train_step += args.train_frequency

            while next_target_step <= global_step:
                for target_parameter, parameter in zip(target_network.parameters(), q_network.parameters()):
                    target_parameter.data.copy_(
                        args.tau * parameter.data + (1.0 - args.tau) * target_parameter.data
                    )
                next_target_step += args.target_network_frequency

            while len(episode_rows) >= next_report_episode:
                steps_per_second = int(global_step / (time.time() - start_time))
                report_rows = episode_rows[
                    next_report_episode - config.DYNAMIC_REPORT_EPISODES:next_report_episode
                ]
                mean_episode_return = float(np.mean([row["return"] for row in report_rows]))
                mean_episode_length = float(np.mean([row["length"] for row in report_rows]))
                print(
                    f"step={global_step} td_loss={last_loss if last_loss is not None else float('nan'):.4f} "
                    f"q_values={last_q_value if last_q_value is not None else float('nan'):.4f} "
                    f"episodes={next_report_episode} mean_ep_return={mean_episode_return:.4f} "
                    f"mean_ep_length={mean_episode_length:.1f} SPS={steps_per_second}"
                )
                if last_loss is not None:
                    writer.add_scalar("losses/td_loss", last_loss, global_step)
                    writer.add_scalar("losses/q_values", last_q_value, global_step)
                writer.add_scalar("charts/SPS", steps_per_second, global_step)
                next_report_episode += config.DYNAMIC_REPORT_EPISODES

            if global_step >= next_checkpoint_step:
                checkpoint_path = save_training_checkpoint(
                    q_network,
                    model_path,
                    global_step,
                    (
                        (episode_rows, log_path),
                        (update_rows, update_log_path),
                    ),
                    model_metadata,
                )
                print(f"Saved dynamic DQN checkpoint to {checkpoint_path}")
                while next_checkpoint_step <= global_step:
                    next_checkpoint_step += config.DYNAMIC_CHECKPOINT_INTERVAL
    finally:
        env.close()
        writer.close()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    save_model_artifact(q_network, model_path, model_metadata)
    pd.DataFrame(episode_rows).to_csv(log_path, index=False)
    pd.DataFrame(update_rows).to_csv(update_log_path, index=False)
    print(f"Saved dynamic DQN model to {model_path}")
    return model_path


if __name__ == "__main__":
    train(tyro.cli(Args))
