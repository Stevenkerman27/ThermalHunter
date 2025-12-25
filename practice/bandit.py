import numpy as np
import matplotlib.pyplot as plt

# --- 1. 参数设置 ---
scale = 0.4        # 噪声强度
BASE_PROBS = [0.4, 0.6] # 基准概率
N_ARMS = len(BASE_PROBS)

EPISODES = 100
ALPHA_START = 0.1
ALPHA_END = 0.01

EPSILON_START = 1
EPSILON_MIN = 0.2

N_RUNS = 100  # <--- 新增：设定重复实验次数

# --- 2. 初始化存储容器 ---
# 我们需要存储所有实验的所有步数数据
# 维度: [实验次数, Episode数+1, 动作数]
all_q_histories = np.zeros((N_RUNS, EPISODES + 1, N_ARMS))

# --- 3. 循环 N 次实验 ---
print(f"开始运行 {N_RUNS} 次独立实验...")

for run in range(N_RUNS):
    # 每次实验都要重置 Q-table
    q_table = np.zeros(N_ARMS)
    q_history = np.zeros((EPISODES + 1, N_ARMS))
    
    # 训练循环
    for i in range(EPISODES):
        # 计算衰减
        progress = i / EPISODES 
        epsilon = EPSILON_START - progress * (EPSILON_START - EPSILON_MIN)
        ALPHA = ALPHA_START - progress * (ALPHA_START - ALPHA_END)
        epsilon = max(EPSILON_MIN, epsilon)

        # [生成环境噪声]
        current_noise = np.random.normal(0, 1, N_ARMS) 
        current_probs = [
            np.clip(base + scale * noise, 0.0, 1.0) 
            for base, noise in zip(BASE_PROBS, current_noise)
        ]

        # 动作选择
        if np.random.random() < epsilon:
            action = np.random.randint(N_ARMS)
        else:
            action = np.argmax(q_table) # 简单的 argmax 会有 ties 问题，但这里先保持原样

        
        # 获取奖励
        reward = 1 if np.random.random() < current_probs[action] else 0
        
        # 更新 Q-Table
        current_q = q_table[action]
        q_table[action] = current_q + ALPHA * (reward - current_q)
        
        q_history[i+1] = q_table.copy()
    
    # 将本次实验的历史记录存入总容器
    all_q_histories[run] = q_history

# --- 4. 数据统计处理 ---
# 计算 N 次实验的平均值 (Mean) 和 标准差 (Std)
# axis=0 表示沿着“实验次数”这个维度压缩
mean_q_history = np.mean(all_q_histories, axis=0) # Shape: [EPISODES+1, N_ARMS]
std_q_history = np.std(all_q_histories, axis=0)   # Shape: [EPISODES+1, N_ARMS]

# --- 5. 可视化 ---
plt.figure(figsize=(12, 6))
colors = ['green', 'orange', 'red']
labels = [f'Action {i+1} (Base Reward: {prob})' for i, prob in enumerate(BASE_PROBS)]

for arm in range(N_ARMS):    
    # 画出 Q值的平均曲线 (实线)
    plt.plot(mean_q_history[:, arm], label=f"{labels[arm]}", color=colors[arm], linewidth=2.5)
    
    # 画出标准差阴影 (核心改动：展示波动范围/严谨性)
    plt.fill_between(
        range(EPISODES + 1), 
        mean_q_history[:, arm] - std_q_history[:, arm], 
        mean_q_history[:, arm] + std_q_history[:, arm], 
        color=colors[arm], alpha=0.2
    )
    
    # D. 画出基准线
    plt.axhline(y=BASE_PROBS[arm], color=colors[arm], linestyle='--', alpha=0.6)

plt.title(f'Q-Learning Averaged over {N_RUNS} Runs', fontsize=14)
plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Probability / Q-Value', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)

plt.show()

from scipy.stats import norm

# --- 6. 打印最终结果与统计显著性概率 ---
print(f"\n=== {N_RUNS} 次实验后的统计分析 ===")

# 1. 获取最终时刻所有实验的 Q 值 [N_RUNS, N_ARMS]
final_q_values = all_q_histories[:, -1, :]

# 2. 找到理论上的最优和次优动作索引
sorted_indices = np.argsort(BASE_PROBS)
best_idx = sorted_indices[-1]      # 最优动作索引 (0.4)
second_idx = sorted_indices[-2]    # 次优动作索引 (0.35)

# 3. 计算这两者的均值和方差
mu_best = np.mean(final_q_values[:, best_idx])
var_best = np.var(final_q_values[:, best_idx], ddof=1) # ddof=1 使用样本方差

mu_second = np.mean(final_q_values[:, second_idx])
var_second = np.var(final_q_values[:, second_idx], ddof=1)

print(f"理论最优 Action {best_idx+1}: Mean={mu_best:.4f}, Var={var_best:.4f}")
print(f"理论次优 Action {second_idx+1}: Mean={mu_second:.4f}, Var={var_second:.4f}")

# 4. 计算差异分布 (Diff = Q_best - Q_second)
# 我们假设 Q 值服从正态分布，则差异也服从正态分布
mu_diff = mu_best - mu_second
sigma_diff = np.sqrt(var_best + var_second)

# 5. 计算 Z-score 和 胜出概率
# 我们想知道 P(Diff > 0) 的概率
if sigma_diff == 0:
    # 极罕见情况：方差为0（比如epsilon=0且没有噪声），直接比较均值
    prob_success = 1.0 if mu_diff > 0 else 0.0
else:
    z_score = mu_diff / sigma_diff
    prob_success = norm.cdf(z_score)

print("-" * 30)
print(f"统计信度 (Confidence): {prob_success*100:.2f}%")
print(f"含义: 在当前的超参数和噪声下，有 {prob_success*100:.2f}% 的概率")
print(f"      训练出的 Agent 认为 Action {best_idx+1} 优于 Action {second_idx+1}。")