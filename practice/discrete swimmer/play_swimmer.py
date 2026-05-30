import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import imageio
import time 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================= 1. 必须重新注册环境 =================
if "GridSwimmer-v0" in gym.envs.registry:
    del gym.envs.registry["GridSwimmer-v0"]

register(
    id="GridSwimmer-v0",
    entry_point="swimmer:GridSwimmerEnv", 
    max_episode_steps=100,
)

# ================= 2. 加载模型与配置 =================
GRID_SIZE = 25
MODEL_FILE = "practice/discrete swimmer/my_q_table.npy"

try:
    q_table = np.load(MODEL_FILE)
    print(f"成功加载模型: {MODEL_FILE}")
except FileNotFoundError:
    print(f"错误: 找不到 {MODEL_FILE}，请先运行训练脚本！")
    exit()

# ================= 3. 播放演示 =================
def play():
    env = gym.make("GridSwimmer-v0", grid_size=GRID_SIZE, render_mode="human")

    for episode in range(10):
        obs, info = env.reset()
        state = tuple(obs)
        done = False
        total_reward = 0

        print(f"=== Episode {episode + 1} start ===")

        while not done:
            env.render()
            time.sleep(0.05) 

            action = np.argmax(q_table[state])
            obs, reward, terminated, truncated, info = env.step(action)
            state = tuple(obs)
            total_reward += reward

            done = terminated or truncated

        print(f"Episode {episode + 1} end, reward: {total_reward}")
        time.sleep(1)

    env.close()

def save_as_gif(filename="swimmer_demo.gif", num_episodes=1):
    env = gym.make("GridSwimmer-v0", grid_size=GRID_SIZE, render_mode="rgb_array")

    frames = [] 

    for episode in range(num_episodes):
        obs, info = env.reset()
        state = tuple(obs)
        done = False

        print(f"Recording Episode {episode + 1}...")

        while not done:
            frame = env.render()
            frames.append(frame)

            action = np.argmax(q_table[state])
            obs, reward, terminated, truncated, info = env.step(action)
            state = tuple(obs)

            done = terminated or truncated

    env.close()

    print(f"Saving GIF to {filename}...")
    imageio.mimsave(filename, frames, fps=10, loop=0)
    print("Done!")

def visualize_trajectories(filename="practice/discrete swimmer/swimmer_trajectories.png", num_episodes=10):
    env = gym.make("GridSwimmer-v0", grid_size=GRID_SIZE)
    fig, ax = plt.subplots(figsize=(10, 10)) # 稍微增大画布以容纳大字体

    # 设置坐标轴
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    
    # 主要刻度每5个一个标签
    ax.set_xticks(np.arange(0, GRID_SIZE + 1, 5))
    ax.set_yticks(np.arange(0, GRID_SIZE + 1, 5))
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # 次要刻度每个格一个，用于画细网格线
    ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1), minor=True)
    ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1), minor=True)
    
    # 绘制网格
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
    ax.set_aspect('equal')

    # 颜色生成器
    cmap = plt.colormaps.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(num_episodes)]

    for i in range(num_episodes):
        obs, info = env.reset()
        state = tuple(obs)

        # 获取初始位置和目标位置
        # 注意：env 是 TimeLimit 包装器，需要访问 unwrapped
        start_pos = env.unwrapped._agent_pos.copy()
        target_pos = env.unwrapped._target_pos.copy()

        path = [start_pos.copy()]
        done = False

        while not done:
            action = np.argmax(q_table[state])
            obs, reward, terminated, truncated, info = env.step(action)
            state = tuple(obs)
            path.append(env.unwrapped._agent_pos.copy())
            done = terminated or truncated

        # 绘制轨迹 (涂色)
        color = colors[i]
        for step_pos in path:
            rect = patches.Rectangle((step_pos[0], step_pos[1]), 1, 1, 
                                     linewidth=0, edgecolor='none', facecolor=color, alpha=0.3)
            ax.add_patch(rect)

        # 绘制起点和终点
        ax.scatter(start_pos[0] + 0.5, start_pos[1] + 0.5, color=color, marker='o', s=100, edgecolors='black', label=f'Ep {i+1} Start' if i < 3 else "")
        ax.scatter(target_pos[0] + 0.5, target_pos[1] + 0.5, color=color, marker='x', s=100, linewidths=3, label=f'Ep {i+1} Target' if i < 3 else "")

    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(filename, bbox_inches='tight')
    print(f"Trajectories saved to {filename}")
    plt.close()

if __name__ == "__main__":
    # play()
    # save_as_gif("play_swimmer.gif", num_episodes=8)
    visualize_trajectories(num_episodes=10)