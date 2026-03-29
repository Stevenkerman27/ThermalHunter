import numpy as np
import gymnasium as gym
from glider_discrete_simp import GliderEnv
import glob
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- 配置区 ---
N_EPISODES = 500  
POLAR_BASE = "glider"
Q_TABLE_DIR = "q_table"
#Q_TABLE_PATH = os.path.join(Q_TABLE_DIR, "q_table_v0.pkl")
Q_TABLE_PATH = os.path.join(Q_TABLE_DIR, "q_table_high.pkl")

with open(Q_TABLE_PATH, "rb") as f:
    q_table = pickle.load(f)

def run_eval(env, policy_type="random"):
    all_rewards = []
    all_climb_heights = [] # 记录爬升高度
    for ep in range(N_EPISODES):
        # 从 reset 的 info 中获取初始高度
        state, info = env.reset(options={"resettime": 80})
        h_start = info["height"]
        
        ep_reward = 0
        done = False
        last_height = h_start
        
        while not done:
            action = env.action_space.sample() if policy_type == "random" else np.argmax(q_table[tuple(state)])
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            last_height = info["height"] # 从 step 的 info 中获取实时高度
            done = terminated or truncated
            
        all_rewards.append(ep_reward)
        all_climb_heights.append(last_height - h_start) # 计算本集总爬升
        
    return all_rewards, all_climb_heights

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wind_dir = os.path.join(base_dir, 'wind')
    h5_files = sorted(glob.glob(os.path.join(wind_dir, 'snapshots_s*.h5')))

    env = GliderEnv(h5_file_path=h5_files, polar_file_base=POLAR_BASE)

    # 获取数据
    rnd_rewards, rnd_climbs = run_eval(env, "random")
    exp_rewards, exp_climbs = run_eval(env, "expert")
    
    # 打印 Reward 结果（保留原功能）
    print(f"--- Reward 统计 ---")
    print(f"随机策略: Mean={np.mean(rnd_rewards):.2f}, Std={np.std(rnd_rewards):.2f}")
    print(f"专家策略: Mean={np.mean(exp_rewards):.2f}, Std={np.std(exp_rewards):.2f}")

    # 整理爬升高度数据用于绘图
    df = pd.DataFrame({
        'Climb': rnd_climbs + exp_climbs,
        'Policy': ['Random'] * N_EPISODES + ['Strategy'] * N_EPISODES
    })

    # --- 使用 Seaborn 可视化 ---
    plt.figure(figsize=(7, 7))
    sns.set_style("white")
    
    my_colors = {"Random": "#7f7f7f", "Strategy": "#d62728"}

    # 1. 散点图
    ax = sns.stripplot(data=df, x='Policy', y='Climb', palette=my_colors, 
                      hue='Policy', jitter=False, alpha=0.4, size=4, legend=False)

    # 2. 箱线图
    sns.boxplot(data=df, x='Policy', y='Climb', width=0.2, 
                showfliers=False, 
                boxprops={'facecolor':'none', 'edgecolor':'black', 'linewidth':1.5},
                medianprops={'color':'#ff7f0e', 'linewidth':2})

    # 3. 在图中直接显示平均值和方差
    policies = ['Random', 'Strategy']
    data_list = [rnd_climbs, exp_climbs]
    
    for i, p in enumerate(policies):
        mean_val = np.mean(data_list[i])
        std_val = np.std(data_list[i])
        
        # 计算文本显示的位置：x 轴索引 i，y 轴放在该组数据的最大值上方一点
        y_pos = max(data_list[i]) + (max(df['Climb']) - min(df['Climb'])) * 0.05
        
        text_str = f"$\mu={mean_val:.1f}$\n$\sigma={std_val:.1f}$"
        plt.text(i, y_pos, text_str, ha='center', va='bottom', 
                 fontsize=10, fontweight='bold', color=my_colors[p])

    # 装饰
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylabel("Episodic Climb Height (m)", fontsize=12)
    plt.xlabel("")
    
    # 画一条 y=0 的水平参考线
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    plt.savefig("trainresult/climb_eval_result.png", dpi=300)
    plt.show()

    env.close()