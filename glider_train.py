import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import glob
import os
import re
import pickle
import time
from eval import run_eval, plot_climb_results
import config

# --- 环境注册 ---
try:
    register(
        id="GliderDiscrete-v0",
        entry_point="glider_discrete_simp:GliderEnv", 
        max_episode_steps=1000,
    )
except:
    pass # 防止重复注册报错

# --- 路径与参数配置 ---
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# 自动搜索 h5 文件
h5_files = sorted(glob.glob(os.path.join(config.WIND_DIR, 'snapshots_s*.h5')), key=natural_key)
if not h5_files:
    raise FileNotFoundError(f"未在 {config.WIND_DIR} 下找到风场 h5 文件")

# --- 实例化环境 ---
env = gym.make(
    "GliderDiscrete-v0", 
    h5_file_path=h5_files, 
    polar_file_base=config.POLAR_BASE,
)

# --- Q-Learning 配置 ---
ALPHA = config.ALPHA
GAMMA = config.GAMMA
EPSILON_START = config.EPSILON_START
EPSILON_END = config.EPSILON_END
EPISODES = config.EPISODES
SAVE_PATH = config.SAVE_PATH
SAVE_INTERVAL = config.SAVE_INTERVAL

# 初始化 Q 表: 状态空间 [7, 3, 3], 动作空间 [3]
q_table = np.zeros((config.BANK_BINS, 3, 3, 3), dtype=np.float32)

epsilon_decay_step = (EPSILON_START - EPSILON_END) / (EPISODES * 0.7)
epsilon = EPSILON_START

def select_action(state, epsilon):
    # state 为 [bank_idx, idx_az, idx_dw]
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return np.argmax(q_table[tuple(state)])

rewards_history = []
climb_history = []

track_indices = config.TRACK_INDICES
q_value_history = {idx: [] for idx in track_indices}
print(f"开始训练... 状态空间: {env.observation_space}, 动作空间: {env.action_space}, 追踪 Q 表索引: {track_indices}")
start_time = time.perf_counter()
total_steps = 0

for episode in range(EPISODES):
    # 随机化起始时间点，总步数 600，预留足够时间完成一个 episode
    random_reset_time = np.random.randint(30, 300)
    state, info = env.reset(options={"resettime": random_reset_time})
    h_start = info["height"]
    total_reward = 0
    done = False
    
    while not done:
        action = select_action(state, epsilon)
        
        # 执行步进
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_steps += 1
        
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
    
    h_end = info["height"]
    climb_history.append(h_end - h_start)
    
    for idx in track_indices:
        # 追踪特定状态下的所有动作 Q 值中的一个，或者修改为追踪特定动作
        # 这里原代码逻辑是 q_table[idx]，idx 为 (s0, s1, s2, a)
        q_value_history[idx].append(q_table[idx])

    rewards_history.append(total_reward)

    # Epsilon 线性衰减
    if epsilon > EPSILON_END:
        epsilon -= epsilon_decay_step
    
    if (episode + 1) % 50 == 0:
        elapsed = time.perf_counter() - start_time
        sps = total_steps / elapsed # 计算每秒运行的步数
        avg_r = np.mean(rewards_history[-50:])
        avg_c = np.mean(climb_history[-50:])
        
        print(f"Ep: {episode+1:4} | Last 50 Avg R: {avg_r:8.2f} | Avg Climb: {avg_c:6.1f}m | Speed: {sps:6.1f} steps/s | Eps: {epsilon:.3f}")
    
    # 每隔 SAVE_INTERVAL 个 episode 保存一次 qtable
    if (episode + 1) % SAVE_INTERVAL == 0:
        save_path = os.path.join(config.Q_TABLE_DIR, f"q_table_E_{episode+1}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(q_table, f)
        print(f"已保存 Q-table 到 {save_path}")

# --- 保存与绘图 ---
with open(SAVE_PATH, "wb") as f:
    pickle.dump(q_table, f)

print(f"训练完成，模型保存至 {SAVE_PATH}")
# --- 自动评估 (利用已加载的环境) ---
print("\n开始自动评估性能...")
N_EVAL_EPISODES = config.N_EVAL_EPISODES 
rnd_rewards, rnd_climbs = run_eval(env, q_table, n_episodes=N_EVAL_EPISODES, policy_type="random")
exp_rewards, exp_climbs = run_eval(env, q_table, n_episodes=N_EVAL_EPISODES, policy_type="trained")

print(f"--- 评估统计 ({N_EVAL_EPISODES} Episodes) ---")
print(f"随机策略: Mean Climb={np.mean(rnd_climbs):.1f}m")
print(f"训练策略: Mean Climb={np.mean(exp_climbs):.1f}m")

env.close()

# 绘制评估图表 (不立即阻塞 show)
eval_plot_path = os.path.join(config.TRAIN_RESULT_DIR, "climb_eval_result.png")
plot_climb_results(rnd_climbs, exp_climbs, save_path=eval_plot_path, show=False)

# --- 绘制训练曲线 ---
fig, (ax1, ax_climb, ax2) = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})

# 子图 1: 奖励曲线
ax1.plot(rewards_history, label='Total Reward', alpha=0.3, color='blue')
if len(rewards_history) >= 50:
    moving_avg = np.convolve(rewards_history, np.ones(50)/50, mode='valid')
    ax1.plot(range(49, len(rewards_history)), moving_avg, label='Moving Average (50)', color='red', linewidth=2)

ax1.set_ylabel('Total Reward')
ax1.set_title('Training Performance (Reward)')
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_ylim(min(rewards_history), max(rewards_history))

# 爬升高度曲线
ax_climb.plot(climb_history, label='Climb Height', alpha=0.3, color='forestgreen')
if len(climb_history) >= 50:
    moving_avg_climb = np.convolve(climb_history, np.ones(50)/50, mode='valid')
    ax_climb.plot(range(49, len(climb_history)), moving_avg_climb, label='Moving Average (50)', color='darkgreen', linewidth=2)

ax_climb.set_ylabel('Climb Height (m)')
ax_climb.set_title('Training Performance (Climb)')
ax_climb.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax_climb.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax_climb.grid(True, linestyle='--', alpha=0.6)
ax_climb.set_ylim(min(climb_history), max(climb_history))

# --- 子图 2: 特定 Q 值变化 ---
for idx, history in q_value_history.items():
    label_str = f"$Q(s:{idx[:2]}, a:{idx[2]})$"
    ax2.plot(history, label=label_str)

ax2.set_xlabel('Episode')
ax2.set_ylabel('$Q$ Value')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
train_plot_path = os.path.join(config.TRAIN_RESULT_DIR, "train_result.png")
plt.savefig(train_plot_path, dpi=300)
plt.show()