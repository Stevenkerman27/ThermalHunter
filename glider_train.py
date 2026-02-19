import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import glob
import os
import re
import pickle

# --- 环境注册 ---
# 确保 entry_point 与你的文件名 glider_discrete_simp.py 一致
try:
    register(
        id="GliderDiscrete-v0",
        entry_point="glider_discrete_simp:GliderEnv", 
        max_episode_steps=2000,
    )
except:
    pass # 防止重复注册报错

# --- 路径与参数配置 ---
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

base_dir = os.path.dirname(os.path.abspath(__file__))
wind_dir = os.path.join(base_dir, 'wind')

# 自动搜索 h5 文件
h5_files = sorted(glob.glob(os.path.join(wind_dir, 'snapshots_s*.h5')), key=natural_key)
if not h5_files:
    raise FileNotFoundError(f"未在 {wind_dir} 下找到风场 h5 文件")

polar_base = "glider"

# --- 实例化环境 ---
env = gym.make(
    "GliderDiscrete-v0", 
    h5_file_path=h5_files, 
    polar_file_base=polar_base,
)

# --- Q-Learning 配置 ---
ALPHA = 0.01
GAMMA = 0.98
EPSILON_START = 1.0
EPSILON_END = 0.2
EPISODES = 10000
SAVE_PATH = "q_table_v0.pkl"

# 初始化 Q 表: 状态空间 [3, 3], 动作空间 [9]
q_table = np.full((3, 3, 9), 0, dtype=np.float32)

epsilon_decay_step = (EPSILON_START - EPSILON_END) / EPISODES
epsilon = EPSILON_START

def select_action(state, epsilon):
    # state 为 [idx_az, idx_dw]
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return np.argmax(q_table[tuple(state)])

rewards_history = []

track_indices = [(0, 0, 8), (0, 1, 7), (2,2,0), (2, 0, 2)] 
q_value_history = {idx: [] for idx in track_indices}
print(f"开始训练... 状态空间: {env.observation_space}, 动作空间: {env.action_space}, 追踪 Q 表索引: {track_indices}")

for episode in range(EPISODES):
    # 环境 reset 返回 (obs, info)
    state, _ = env.reset(options={"resettime": 80})
    total_reward = 0
    done = False
    
    while not done:
        action = select_action(state, epsilon)
        
        # 执行步进，返回 5 个值
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 转换为 tuple 以索引 numpy 数组
        s_idx = tuple(state)
        ns_idx = tuple(next_state)
        
        # Q-Table 更新 (Temporal Difference)
        if terminated:
            td_target = reward
        else:
            td_target = reward + GAMMA * np.max(q_table[ns_idx])
            
        q_table[s_idx][action] += ALPHA * (td_target - q_table[s_idx][action])
        
        state = next_state
        total_reward += reward
    for idx in track_indices:
        q_value_history[idx].append(q_table[idx])

    rewards_history.append(total_reward)

    # Epsilon 线性衰减
    if epsilon > EPSILON_END:
        epsilon -= epsilon_decay_step
    
    if (episode + 1) % 50 == 0:
        avg_r = np.mean(rewards_history[-50:])
        print(f"Ep: {episode+1:4} | Last 50 Avg Reward: {avg_r:8.2f} | Eps: {epsilon:.3f}")

# --- 保存与绘图 ---
with open(SAVE_PATH, "wb") as f:
    pickle.dump(q_table, f)

print(f"训练完成，模型保存至 {SAVE_PATH}")
env.close()

# 创建包含 2 个子图的画布，共享 X 轴（Episode）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# --- 子图 1: 奖励曲线 ---
ax1.plot(rewards_history, label='Total Reward', alpha=0.3, color='blue')
# 绘制滑动平均线
if len(rewards_history) >= 50:
    moving_avg = np.convolve(rewards_history, np.ones(50)/50, mode='valid')
    ax1.plot(range(49, len(rewards_history)), moving_avg, label='Moving Average (50)', color='red')
ax1.set_ylabel('Total Reward')
ax1.set_title('Training Performance & Q-Value Convergence')
ax1.set_ylim(min(rewards_history), max(rewards_history))
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.6)

# --- 子图 2: 特定 Q 值变化 ---
for idx, history in q_value_history.items():
    # 使用 LaTeX 格式标记 Q(s, a)
    label_str = f"$Q(s:{idx[:2]}, a:{idx[2]})$"
    ax2.plot(history, label=label_str)

ax2.set_xlabel('Episode')
ax2.set_ylabel('$Q$ Value')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.6)

# 自动调整布局防止标签重叠
plt.tight_layout()
plt.show()