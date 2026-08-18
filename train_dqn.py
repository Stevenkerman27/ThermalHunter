# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
import json
import glob
from dataclasses import dataclass
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import config
from glider_discrete_simp import GliderEnv


class ReplayBuffer:
    def __init__(self, capacity, observation_space, action_space, device):
        self.capacity = capacity
        self.device = device
        self.observations = np.empty((capacity, *observation_space.shape), dtype=np.float32)
        self.next_observations = np.empty((capacity, *observation_space.shape), dtype=np.float32)
        self.actions = np.empty((capacity,), dtype=np.int64)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, observations, next_observations, actions, rewards, dones):
        for index in range(len(actions)):
            self.observations[self.position] = observations[index]
            self.next_observations[self.position] = next_observations[index]
            self.actions[self.position] = actions[index]
            self.rewards[self.position] = rewards[index]
            self.dones[self.position] = dones[index]
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            raise ValueError("replay buffer does not contain a full batch")
        indices = np.random.randint(0, self.size, size=batch_size)
        return SimpleNamespace(
            observations=torch.as_tensor(self.observations[indices], device=self.device),
            next_observations=torch.as_tensor(self.next_observations[indices], device=self.device),
            actions=torch.as_tensor(self.actions[indices], device=self.device).reshape(-1, 1),
            rewards=torch.as_tensor(self.rewards[indices], device=self.device),
            dones=torch.as_tensor(self.dones[indices], device=self.device),
        )

# --- Environment Registration ---
try:
    h5_files = sorted(glob.glob(os.path.join(config.WIND_DIR, 'snapshots_s*.h5')), key=config.natural_key)
    gym.register(
        id="GliderContinuous-v0",
        entry_point="glider_discrete_simp:GliderEnv", 
        max_episode_steps=1000,
        kwargs={
            "h5_file_path": h5_files,
            "polar_file_base": config.POLAR_BASE,
            "continuous_obs": True
        }
    )
except Exception:
    pass 

def normalize_state(state, sensor_stats):
    s = state.copy().astype(np.float32)
    # aoa_idx, bank_idx normalization to ~[-1, 1]
    s[0] = (s[0] - (config.AOA_BINS / 2)) / (config.AOA_BINS / 2)
    s[1] = (s[1] - (config.BANK_BINS / 2)) / (config.BANK_BINS / 2)
    
    s[2] = (s[2] - sensor_stats["w_accel"]["mean"]) / sensor_stats["w_accel"]["std"]
    s[3] = (s[3] - sensor_stats["delta_w"]["mean"]) / sensor_stats["delta_w"]["std"]
    return s

def load_sensor_stats(stats_path=None):
    from analyze_bins import sensor_stats_path

    stats_path = sensor_stats_path() if stats_path is None else stats_path
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"sensor statistics not found: {stats_path}")
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)

class GliderWrapper(gym.Wrapper):
    def __init__(self, env, sensor_stats_path=None):
        super().__init__(env)
        self.sensor_stats = load_sensor_stats(sensor_stats_path)
        
        # Define the observation space after normalization
        # Obs: [aoa, bank, w_accel, delta_w]
        low = np.array([-2.0, -2.0, -10.0, -10.0], dtype=np.float32)
        high = np.array([2.0, 2.0, 10.0, 10.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        if kwargs.get("options") is None:
            kwargs["options"] = {}
        # Inject random resettime if not provided
        if "resettime" not in kwargs["options"]:
            kwargs["options"]["resettime"] = config.sample_start_frame(np.random)
        obs, info = self.env.reset(**kwargs)
        return normalize_state(obs, self.sensor_stats), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return normalize_state(obs, self.sensor_stats), reward, terminated, truncated, info

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = config.SEED
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    load_model: str = ""
    """path to an existing model to load (empty for none)"""
    num_checkpoints: int = 5
    """the number of checkpoints to save during training"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "GliderContinuous-v0"
    """the id of the environment"""
    total_timesteps: int = config.DQN_TOTAL_TIMESTEPS
    """total timesteps of the experiments"""
    learning_rate: float = config.DQN_LR
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = config.DQN_BUFFER_SIZE
    """the replay memory buffer size"""
    gamma: float = config.DQN_GAMMA
    """the discount factor gamma"""
    tau: float = config.DQN_TAU
    """the target network update rate"""
    target_network_frequency: int = config.DQN_TARGET_FREQ
    """the timesteps it takes to update the target network"""
    batch_size: int = config.DQN_BATCH_SIZE
    """the batch size of sample from the reply memory"""
    start_e: float = config.DQN_EPSILON_START
    """the starting epsilon for exploration"""
    end_e: float = config.DQN_EPSILON_END
    """the ending epsilon for exploration"""
    exploration_fraction: float = config.DQN_EXPLORATION_FRACTION
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = config.DQN_LEARNING_STARTS
    """timestep to start learning"""
    train_frequency: int = config.DQN_TRAIN_FREQ
    """the frequency of training"""
    sensor_stats_episodes: int = config.SENSOR_STATS_EPISODES
    """random episodes used to build sensor normalization statistics"""
    reward_w_accel: float = config.REWARD_W_ACCEL_WEIGHT
    """vertical wind acceleration reward coefficient"""
    model_path: str = ""
    """final model path; empty uses the default path"""
    log_path: str = ""
    """training CSV path; empty uses the default path"""
    sensor_stats_path: str = ""
    """sensor-statistics path; empty uses the default path"""
    training_plot_path: str = ""
    """training-plot path; empty uses the default path"""


def make_env(env_id, seed, idx, capture_video, run_name, reward_w_accel, sensor_stats_path, memory_mode=False):
    def thunk():
        env = gym.make(env_id, memory_mode=memory_mode, reward_w_accel=reward_w_accel)
        
        env = GliderWrapper(env, sensor_stats_path)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, observation_shape, action_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(observation_shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_count),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    if args.reward_w_accel <= 0:
        raise ValueError("reward_w_accel must be positive")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_num_threads(config.DQN_TORCH_THREADS)

    from analyze_bins import collect_sensor_stats, sensor_stats_path as default_sensor_stats_path
    sensor_stats_path = args.sensor_stats_path or default_sensor_stats_path()
    collect_sensor_stats(args.sensor_stats_episodes, sensor_stats_path)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + i,
                i,
                args.capture_video,
                run_name,
                args.reward_w_accel,
                sensor_stats_path,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs.single_observation_space.shape, envs.single_action_space.n).to(device)
    
    # Load model if specified
    if args.load_model and os.path.exists(args.load_model):
        q_network.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Loaded existing model from {args.load_model}")
    elif os.path.exists(config.DQN_SAVE_PATH):
        # Optional: Auto-load global model if found? Let's make it explicit via args instead.
        pass

    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs.single_observation_space.shape, envs.single_action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
    )
    start_time = time.time()

    import csv
    log_file_path = args.log_path or os.path.join(config.TRAIN_RESULT_DIR, "dqn_train_stats.csv")
    log_file = open(log_file_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "return", "climb"])

    recent_returns = []
    recent_climbs = []
    # Full history for plotting
    all_returns = []
    all_climbs = []
    
    # Track initial height for each env to calculate net climb
    # IMPORTANT: Initial heights MUST be captured after the first reset
    episode_start_heights = np.zeros(envs.num_envs)

    checkpoint_interval = args.total_timesteps // args.num_checkpoints if args.num_checkpoints > 0 else None
    
    # TRY NOT TO MODIFY: start the game
    obs, infos = envs.reset(seed=args.seed)
    episode_start_heights = infos["height"].copy()
    
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        # If model was loaded, maybe we want to start with lower epsilon? 
        # For now, follow the schedule.
        
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "_episode" in infos:
            for idx, d in enumerate(infos["_episode"]):
                if d:
                    # After auto-reset, GliderEnv provides 'prev_height' (terminal height of finished episode)
                    # and 'height' (start height of new episode)
                    final_height = infos.get("prev_height", [0]*envs.num_envs)[idx]
                    net_climb = final_height - episode_start_heights[idx]
                    ep_return = infos["episode"]["r"][idx]
                    
                    recent_returns.append(ep_return)
                    recent_climbs.append(net_climb)
                    all_returns.append(ep_return)
                    all_climbs.append(net_climb)

                    # Log to CSV
                    log_writer.writerow([global_step, ep_return, net_climb])
                    log_file.flush()

                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", infos["episode"]["l"][idx], global_step)
                    writer.add_scalar("charts/net_climb", net_climb, global_step)

                    if len(recent_climbs) >= 10:
                        avg_return = np.mean(recent_returns)
                        avg_climb = np.mean(recent_climbs)
                        print(f"global_step={global_step}, avg_return (last 10 eps)={avg_return:.2f}, avg_climb={avg_climb:.1f}m")
                        recent_returns.clear()
                        recent_climbs.clear()
                    
                    # Update initial height for the NEXT episode in this env
                    # After reset, infos["height"] contains the height of the new episode
                    episode_start_heights[idx] = infos["height"][idx]

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                final_obs = infos.get("final_observation")
                if final_obs is not None:
                    real_next_obs[idx] = final_obs[idx]
        
        rb.add(obs, real_next_obs, actions, rewards, np.logical_or(terminations, truncations))

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 1000 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

        # Checkpoint saving
        if args.save_model and checkpoint_interval is not None:
            if (global_step + 1) % checkpoint_interval == 0 or (global_step + 1) == args.total_timesteps:
                model_path = f"runs/{run_name}/{args.exp_name}_{global_step + 1}.cleanrl_model"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(q_network.state_dict(), model_path)
                print(f"Checkpoint saved to {model_path} at step {global_step + 1}")
                
                # Additionally save the final model to the global path
                if (global_step + 1) == args.total_timesteps:
                    final_model_path = args.model_path or config.DQN_SAVE_PATH
                    torch.save(q_network.state_dict(), final_model_path)
                    print(f"Final model saved to: {final_model_path}")

    log_file.close()
    envs.close()
    del rb
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    writer.close()

    # --- Plotting Training Curve (External Script) ---
    if all_climbs:
        from plot_dqn_train import plot_dqn_training
        plot_dqn_training(log_file_path, args.training_plot_path or None)
    else:
        print("No completed episodes; skipped DQN training plot.")

