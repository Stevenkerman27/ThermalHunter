import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import glob
import os
import random
import time
import json
import matplotlib.pyplot as plt
from collections import deque
import config
from eval import plot_climb_results

# --- Environment Registration ---
try:
    register(
        id="GliderContinuous-v0",
        entry_point="glider_discrete_simp:GliderEnv", 
        max_episode_steps=1000,
    )
except:
    pass 

# --- DQN Model ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=config.DQN_HIDDEN_SIZE):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32), 
                np.array(next_state), np.array(done, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)

# --- Normalization Helper ---
# 加载统计数据
stats_path = os.path.join(config.BASE_DIR, "sensor_stats.json")
if os.path.exists(stats_path):
    with open(stats_path, "r") as f:
        SENSOR_STATS = json.load(f)
    print(f"Loaded sensor statistics for normalization from {stats_path}")
else:
    SENSOR_STATS = None
    print("Warning: sensor_stats.json not found. Using fallback normalization.")

def normalize_state(state):
    # state: [aoa_idx, bank_idx, w_accel, delta_w]
    s = state.copy().astype(np.float32)
    # 离散索引归一化到 [-1, 1]
    s[0] = (s[0] - (config.AOA_BINS / 2)) / (config.AOA_BINS / 2)
    s[1] = (s[1] - (config.BANK_BINS / 2)) / (config.BANK_BINS / 2)
    
    # 连续传感器值进行 Z-score 归一化 (x - mean) / std
    if SENSOR_STATS:
        s[2] = (s[2] - SENSOR_STATS["w_accel"]["mean"]) / SENSOR_STATS["w_accel"]["std"]
        s[3] = (s[3] - SENSOR_STATS["delta_w"]["mean"]) / SENSOR_STATS["delta_w"]["std"]
    else:
        # Fallback
        s[2] = s[2] * 2.0 
        s[3] = s[3] * 5.0
    return s

# --- Training Script ---
def train():
    h5_files = sorted(glob.glob(os.path.join(config.WIND_DIR, 'snapshots_s*.h5')), key=config.natural_key)
    env = gym.make("GliderContinuous-v0", h5_file_path=h5_files, polar_file_base=config.POLAR_BASE, continuous_obs=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=config.DQN_LR)
    buffer = ReplayBuffer(config.DQN_BUFFER_SIZE)
    
    epsilon = config.DQN_EPSILON_START
    epsilon_decay = (config.DQN_EPSILON_START - config.DQN_EPSILON_END) / (config.DQN_EPISODES * 0.8)
    
    rewards_history = []
    climb_history = []
    
    print(f"Starting DQN Training on {device}...")
    start_time = time.perf_counter()
    total_steps = 0

    for ep in range(config.DQN_EPISODES):
        random_reset_time = np.random.randint(config.RESET_TIME_MIN, config.RESET_TIME_MAX)
        state, info = env.reset(options={"resettime": random_reset_time})
        state = normalize_state(state)
        h_start = info["height"]
        ep_reward = 0
        
        while True:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = q_net(state_t).argmax().item()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = normalize_state(next_state)
            done = terminated or truncated
            
            buffer.push(state, action, reward, next_state, terminated) # use terminated for TD target
            
            state = next_state
            ep_reward += reward
            total_steps += 1
            
            # Optimization step
            if len(buffer) > config.DQN_BATCH_SIZE:
                b_state, b_action, b_reward, b_next_state, b_done = buffer.sample(config.DQN_BATCH_SIZE)
                
                b_state = torch.FloatTensor(b_state).to(device)
                b_action = torch.LongTensor(b_action).unsqueeze(1).to(device)
                b_reward = torch.FloatTensor(b_reward).unsqueeze(1).to(device)
                b_next_state = torch.FloatTensor(b_next_state).to(device)
                b_done = torch.FloatTensor(b_done).unsqueeze(1).to(device)
                
                q_values = q_net(b_state).gather(1, b_action)
                with torch.no_grad():
                    max_next_q = target_net(b_next_state).max(1)[0].unsqueeze(1)
                    target_q = b_reward + (config.DQN_GAMMA * max_next_q * (1 - b_done))
                
                loss = F.mse_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        rewards_history.append(ep_reward)
        climb_history.append(info["height"] - h_start)
        
        if epsilon > config.DQN_EPSILON_END:
            epsilon -= epsilon_decay
            
        if (ep + 1) % config.DQN_TARGET_UPDATE_INTERVAL == 0:
            target_net.load_state_dict(q_net.state_dict())
            
        if (ep + 1) % 50 == 0:
            elapsed = time.perf_counter() - start_time
            sps = total_steps / elapsed
            avg_r = np.mean(rewards_history[-50:])
            avg_c = np.mean(climb_history[-50:])
            print(f"Ep: {ep+1:4} | Avg R: {avg_r:8.2f} | Avg Climb: {avg_c:6.1f}m | Speed: {sps:6.1f} steps/s | Eps: {epsilon:.3f}")

        if (ep + 1) % config.SAVE_INTERVAL == 0:
            torch.save(q_net.state_dict(), config.DQN_SAVE_PATH.replace(".pth", f"_E{ep+1}.pth"))
            print(f"Saved checkpoint to {config.DQN_SAVE_PATH}")

    torch.save(q_net.state_dict(), config.DQN_SAVE_PATH)
    print(f"Training finished. Model saved to {config.DQN_SAVE_PATH}")
    
    # Save training curves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 子图 1: 奖励曲线
    ax1.plot(rewards_history, label='Total Reward', alpha=0.3, color='blue')
    if len(rewards_history) >= 50:
        moving_avg_r = np.convolve(rewards_history, np.ones(50)/50, mode='valid')
        ax1.plot(range(49, len(rewards_history)), moving_avg_r, label='Moving Average (50)', color='red', linewidth=2)
    ax1.set_ylabel('Total Reward')
    ax1.set_title('DQN Training Performance (Reward)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 子图 2: 爬升高度曲线
    ax2.plot(climb_history, label='Climb Height', alpha=0.3, color='forestgreen')
    if len(climb_history) >= 50:
        moving_avg_c = np.convolve(climb_history, np.ones(50)/50, mode='valid')
        ax2.plot(range(49, len(climb_history)), moving_avg_c, label='Moving Average (50)', color='darkgreen', linewidth=2)
    ax2.set_ylabel('Climb Height (m)')
    ax2.set_xlabel('Episode')
    ax2.set_title('DQN Training Performance (Climb)')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    train_plot_path = os.path.join(config.TRAIN_RESULT_DIR, "dqn_train_result.png")
    plt.savefig(train_plot_path, dpi=300)
    print(f"Training curves saved to {train_plot_path}")
    
    env.close()

if __name__ == "__main__":
    train()
