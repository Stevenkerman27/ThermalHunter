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
NUM_EPISODES = 50       # 训练总局数
LEARNING_RATE = 0.01
GAMMA = 0.1
EPSILON_START = 0.1
EPSILON_END = 0.01
SAVE_FREQ = 5
Q_INIT_VALUE = 10

# =================主程序=================
if __name__ == "__main__":
    # 创建训练环境
    env = gym.make("GridSwimmer-v0", grid_size=GRID_SIZE)
    
    # Q-Table 形状: (is_up, is_down, is_left, is_right, action)
    q_table_shape = (2, 2, 2, 2, env.action_space.n)
    q_table = np.full(q_table_shape, Q_INIT_VALUE)
    
    # 线性衰减设置
    epsilon_decay_step = (EPSILON_START - EPSILON_END) / int(NUM_EPISODES)
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
        
        # --- 定期保存 Q-Table ---
        save_interval = NUM_EPISODES // SAVE_FREQ
        if episode % save_interval == 0:
            save_path = f"practice/discrete swimmer/q_table_E{episode}.npy"
            np.save(save_path, q_table)
            print(f"定期保存模型: {save_path}")

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1} | Epsilon: {epsilon:.2f} | Avg Reward: {avg_reward:.2f}")

    print("训练完成！")

    # === 绘图代码 ===
    plt.figure(figsize=(13, 6))
    data = np.array(metrics)
    
    # 1. 设置基准线 (1.0 代表完美表现)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Optimal Performance')

    # 绘制保存点的竖线
    save_interval = NUM_EPISODES // SAVE_FREQ
    for i in range(SAVE_FREQ + 1):
        cp_episode = i * save_interval
        if cp_episode <= NUM_EPISODES:
            plt.axvline(x=cp_episode, color='black', linestyle=':', alpha=0.8, 
                        label='Checkpoint' if i == 0 else "")

    # 2. 计算平滑参数
    window_size = max(1, int(0.01 * len(data)))
    do_smooth = len(data) >= window_size and window_size > 1

    # 3. 绘制原始数据
    if do_smooth:
        # 轴标题已经指定的数据不使用legend
        plt.plot(data, color='gray', alpha=0.2)
        # 绘制平滑曲线（核心）
        smooth_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        x_smooth = np.arange(window_size, len(data) + 1)
        plt.plot(x_smooth, smooth_data, color='green', linewidth=2)
        plt.ylim(bottom=min(smooth_data) * 0.9, top=1.2)
    else:
        # 轴标题已经指定的数据不使用legend
        plt.plot(data, color='green', alpha=0.6)
        plt.ylim(bottom=min(data) * 0.9, top=1.2)
    
    # 必须为所有文字指定字体大小，xy轴必须有轴标题
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Metrics", fontsize=18)
    plt.title("Swimmer Training Metrics", fontsize=20)

    plt.legend(loc='lower right', fontsize=18)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    plt.tight_layout() # 自动调整布局
    
    # 保存训练结果图
    train_plot_path = "practice/discrete swimmer/train_metrics.png"
    plt.savefig(train_plot_path, dpi=150)
    print(f"训练结果图已保存至: {train_plot_path}")
    
    plt.show()


    print(f"Q-Table 最大值: {np.max(q_table)}")
    print(f"Q-Table 最小值: {np.min(q_table)}")
    print(f"Q-Table 平均值: {np.average(q_table)}")
    

    # === 保存 Q-Table 到文件 ===
    final_save_path = f"practice/discrete swimmer/q_table_E{NUM_EPISODES}.npy"
    np.save(final_save_path, q_table)
    np.save("practice/discrete swimmer/my_q_table.npy", q_table)
    print(f"训练结束，模型已保存为: {final_save_path} 和 my_q_table.npy")
    
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
    test_env.close()