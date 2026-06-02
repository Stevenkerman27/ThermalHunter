# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
import json
import glob
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer
import config
from glider_discrete_simp import GliderEnv

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

def normalize_state(state, sensor_stats=None):
    s = state.copy().astype(np.float32)
    # aoa_idx, bank_idx normalization to ~[-1, 1]
    s[0] = (s[0] - (config.AOA_BINS / 2)) / (config.AOA_BINS / 2)
    s[1] = (s[1] - (config.BANK_BINS / 2)) / (config.BANK_BINS / 2)
    
    if sensor_stats:
        s[2] = (s[2] - sensor_stats["w_accel"]["mean"]) / sensor_stats["w_accel"]["std"]
        s[3] = (s[3] - sensor_stats["delta_w"]["mean"]) / sensor_stats["delta_w"]["std"]
    else:
        s[2] *= 2.0 
        s[3] *= 5.0
    return s

class GliderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        stats_path = os.path.join(config.BASE_DIR, "sensor_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                self.sensor_stats = json.load(f)
        else:
            self.sensor_stats = None
        
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
            kwargs["options"]["resettime"] = np.random.randint(config.RESET_TIME_MIN, config.RESET_TIME_MAX)
        obs, info = self.env.reset(**kwargs)
        return normalize_state(obs, self.sensor_stats), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return normalize_state(obs, self.sensor_stats), reward, terminated, truncated, info

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
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


def make_env(env_id, seed, idx, capture_video, run_name, memory_mode=True):
    def thunk():
        env = gym.make(env_id, memory_mode=memory_mode)
        
        env = GliderWrapper(env)
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs.single_observation_space.shape, envs.single_action_space.n).to(device)
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

    recent_returns = []
    recent_heights = []

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
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
                    recent_returns.append(infos["episode"]["r"][idx])
                    recent_heights.append(infos["height"][idx])

                    writer.add_scalar("charts/episodic_return", infos["episode"]["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", infos["episode"]["l"][idx], global_step)
                    writer.add_scalar("charts/final_height", infos["height"][idx], global_step)

                    if len(recent_heights) >= 10:
                        avg_return = np.mean(recent_returns)
                        avg_height = np.mean(recent_heights)
                        print(f"global_step={global_step}, avg_return (last 10 eps)={avg_return:.2f}, avg_height={avg_height:.1f}")
                        recent_returns.clear()
                        recent_heights.clear()

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]
            elif trunc and "_final_observation" in infos and infos["_final_observation"][idx]:
                # Handle cases where Gymnasium uses _final_observation mask
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations)

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

    envs.close()
    del rb
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        
        # Also save to config.DQN_SAVE_PATH for compatibility with existing scripts
        torch.save(q_network.state_dict(), config.DQN_SAVE_PATH)
        print(f"model also saved to {config.DQN_SAVE_PATH}")

        # Manual evaluation to ensure memory control
        print("Starting evaluation...")
        q_network.eval()
        eval_env = make_env(args.env_id, args.seed + 100, 0, False, f"{run_name}-eval", memory_mode=False)()
        episodic_returns = []
        for i in range(10):
            obs, _ = eval_env.reset()
            done = False
            episodic_return = 0
            while not done:
                with torch.no_grad():
                    q_values = q_network(torch.Tensor(obs).to(device).unsqueeze(0))
                    action = torch.argmax(q_values, dim=1).item()
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episodic_return += reward
                done = terminated or truncated
            episodic_returns.append(episodic_return)
            print(f"Eval episode {i}: return={episodic_return:.2f}")
            writer.add_scalar("eval/episodic_return", episodic_return, i)
        
        eval_env.close()
        print("Evaluation finished and eval_env closed.")

    writer.close()
