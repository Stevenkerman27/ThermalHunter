import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import imageio
import time # 用来控制播放速度

# ================= 1. 必须重新注册环境 =================
# 因为这是一个新的 Python 进程，必须告诉 gym "GridSwimmer-v0" 是什么
if "GridSwimmer-v0" in gym.envs.registry:
    del gym.envs.registry["GridSwimmer-v0"]

register(
    id="GridSwimmer-v0",
    entry_point="swimmer:GridSwimmerEnv", # 确保 swimmer.py 在同目录下
    max_episode_steps=100,
)

# ================= 2. 加载模型与配置 =================
GRID_SIZE = 25
MODEL_FILE = "my_q_table.npy"

# 加载保存的 Q-Table
try:
    q_table = np.load(MODEL_FILE)
    print(f"成功加载模型: {MODEL_FILE}")
except FileNotFoundError:
    print(f"错误: 找不到 {MODEL_FILE}，请先运行训练脚本！")
    exit()

# ================= 3. 播放演示 =================
def play():
    # render_mode="human" 会弹出窗口显示动画
    env = gym.make("GridSwimmer-v0", grid_size=GRID_SIZE, render_mode="human")
    
    # 播放 5 局
    for episode in range(10):
        obs, info = env.reset()
        state = tuple(obs)
        done = False
        total_reward = 0
        
        print(f"=== Episode {episode + 1} start ===")
        
        while not done:
            # 渲染画面
            env.render()
            time.sleep(0.05) 
            
            # === 直接查表取最大值 (贪婪策略) ===
            action = np.argmax(q_table[state])
            
            obs, reward, terminated, truncated, info = env.step(action)
            state = tuple(obs)
            total_reward += reward
            
            done = terminated or truncated
            
        print(f"Episode {episode + 1} end, reward: {total_reward}")
        time.sleep(1) # 每局结束歇一秒

    env.close()

def save_as_gif(filename="swimmer_demo.gif", num_episodes=1):
    # 关键修改 1: render_mode 必须设为 "rgb_array"
    env = gym.make("GridSwimmer-v0", grid_size=GRID_SIZE, render_mode="rgb_array")
    
    frames = [] # 用于存储画面帧
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = tuple(obs)
        done = False
        
        print(f"Recording Episode {episode + 1}...")
        
        while not done:
            # 关键修改 2: 获取当前帧的像素数据并存入列表
            frame = env.render()
            frames.append(frame)
            
            action = np.argmax(q_table[state])
            obs, reward, terminated, truncated, info = env.step(action)
            state = tuple(obs)
            
            done = terminated or truncated

    env.close()

    # 关键修改 3: 保存为 GIF
    print(f"Saving GIF to {filename}...")
    # fps 控制播放速度，loop=0 表示无限循环
    imageio.mimsave(filename, frames, fps=10, loop=0)
    print("Done!")

if __name__ == "__main__":
    play()
    save_as_gif("play_swimmer.gif", num_episodes=8)