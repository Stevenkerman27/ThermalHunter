import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt

# 注册环境
# 确保没有重复注册
if "GridSwimmer-v0" in gym.envs.registry:
    del gym.envs.registry["GridSwimmer-v0"]

register(
    id="GridSwimmer-v0",
    entry_point="swimmer:GridSwimmerEnv",
    max_episode_steps=100,
)

# =================配置参数=================
GRID_SIZE = 25
NUM_EPISODES = 10000       # 训练总局数
LEARNING_RATE = 0.5
GAMMA = 0.6
EPSILON_START = 1.0
EPSILON_END = 0.1
sessions = 1000

# =================主程序=================
if __name__ == "__main__":
    # 创建训练环境
    env = gym.make("GridSwimmer-v0", grid_size=GRID_SIZE)
    obs_dim = GRID_SIZE
    
    # Q-Table 形状: (dx, dy, action)
    q_table_shape = (obs_dim, obs_dim, env.action_space.n)
    q_table = np.zeros(q_table_shape)
    
    # 线性衰减设置
    decay_duration = int(NUM_EPISODES * 0.7)
    epsilon_decay_step = (EPSILON_START - EPSILON_END) / decay_duration
    epsilon = EPSILON_START
    
    print(f"环境: {GRID_SIZE}x{GRID_SIZE} Grid | Q-Table Size: {q_table.size}")
    
    rewards_history = []
    metrics = []

    # --- 训练循环 ---
    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        max_reward = int(info["max reward"])
        # 将 numpy array 转换为 tuple 以便作为 Q-table 的索引
        state = tuple(obs) 
        
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-Greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                # np.argmax 默认取第一个最大值，如果全是0就会一直选动作0
                values = q_table[state]
                action = np.random.choice(np.flatnonzero(values == values.max()))

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = tuple(next_obs)
            done = terminated or truncated
            
            # Q-Learning 更新
            best_next_q = np.max(q_table[next_state])
            current_q = q_table[state + (action,)]
            
            # Bellman Equation
            new_q = current_q + LEARNING_RATE * (reward + GAMMA * best_next_q - current_q)
            q_table[state + (action,)] = new_q
            
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        metrics.append(total_reward/max_reward)
        
        # Epsilon 衰减
        if epsilon > EPSILON_END:
            epsilon -= epsilon_decay_step
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1} | Epsilon: {epsilon:.2f} | Avg Reward: {avg_reward:.2f}")

    print("训练完成！")

    # === 绘图代码 ===
    plt.figure(figsize=(10, 6))
    data = np.array(metrics)
    
    # 1. 设置基准线 (1.0 代表完美表现)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Optimal (1.0)')

    # 2. 绘制原始数据（调高透明度，作为背景噪点）
    plt.plot(data, color='gray', alpha=0.2, label='Raw Ratio')

    # 3. 绘制平滑曲线（核心）
    window_size = int(0.01*episode)  # 窗口大小
    if len(data) >= window_size:
        # 使用卷积计算滑动平均
        smooth_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        # 对齐X轴
        x_smooth = np.arange(window_size, len(data) + 1)
        plt.plot(x_smooth, smooth_data, color='blue', linewidth=2, label='Smoothed Ratio')

    plt.xlabel("Episode")
    plt.ylabel("Score Ratio (Actual / Max)")
    plt.title("Normalized Training Performance")
    plt.ylim(bottom=-2.0, top=1.2) # 限制Y轴视野，忽略初期极差的表现，聚焦后期
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # === 保存 Q-Table 到文件 ===
    np.save("my_q_table.npy", q_table)
    print("模型已保存为 my_q_table.npy")
    
    # =================测试演示=================
    print("\n开始演示...")
    test_env = gym.make("GridSwimmer-v0", grid_size=GRID_SIZE, render_mode="human")
    
    for _ in range(2):
        obs, _ = test_env.reset()
        state = tuple(obs)
        done = False
        print("Start Episode...")
        
        while not done:
            action = np.argmax(q_table[state])
            obs, reward, terminated, truncated, _ = test_env.step(action)
            state = tuple(obs)
            done = terminated or truncated
            
    test_env.close()