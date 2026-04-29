import numpy as np
import gymnasium as gym
from glider_discrete_simp import GliderEnv
import glob
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import config

def run_eval(env, q_table, n_episodes=500, policy_type="trained"):
    """执行评估循环"""
    all_rewards = []
    all_climb_heights = []
    for ep in range(n_episodes):
        # 使用与训练相同的随机起始时间逻辑
        random_reset_time = np.random.randint(config.RESET_TIME_MIN, config.RESET_TIME_MAX)
        state, info = env.reset(options={"resettime": random_reset_time})
        h_start = info["height"]
        
        ep_reward = 0
        done = False
        last_height = h_start
        
        while not done:
            if policy_type == "random":
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[tuple(state)])
                
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            last_height = info["height"]
            done = terminated or truncated
            
        all_rewards.append(ep_reward)
        all_climb_heights.append(last_height - h_start)
        
    return all_rewards, all_climb_heights

def plot_climb_results(rnd_climbs, exp_climbs, save_path="trainresult/climb_eval_result.png", show=True):
    """绘制爬升高度对比图 (Seaborn 样式)"""
    n_episodes = len(rnd_climbs)
    df = pd.DataFrame({
        'Climb': rnd_climbs + exp_climbs,
        'Policy': ['Random Policy'] * n_episodes + ['Trained Policy'] * n_episodes
    })

    plt.figure(figsize=(7, 7))
    sns.set_style("white")
    my_colors = {"Random Policy": "#7f7f7f", "Trained Policy": "#d62728"}

    # 1. 散点图
    ax = sns.stripplot(data=df, x='Policy', y='Climb', palette=my_colors, 
                      hue='Policy', jitter=False, alpha=0.4, size=4, legend=False)

    # 2. 箱线图
    sns.boxplot(data=df, x='Policy', y='Climb', width=0.2, 
                showfliers=False, 
                boxprops={'facecolor':'none', 'edgecolor':'black', 'linewidth':1.5},
                medianprops={'color':'#ff7f0e', 'linewidth':2})

    # 3. 显示平均值和方差
    policies = ['Random Policy', 'Trained Policy']
    data_list = [rnd_climbs, exp_climbs]

    for i, p in enumerate(policies):
        mean_val = np.mean(data_list[i])
        std_val = np.std(data_list[i])
        y_pos = max(data_list[i]) + (max(df['Climb']) - min(df['Climb'])) * 0.05
        text_str = f"$\mu={mean_val:.1f}$\n$\sigma={std_val:.1f}$"
        plt.text(i, y_pos, text_str, ha='center', va='bottom', 
                 fontsize=16, fontweight='bold', color=my_colors[p])
        
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylabel("Episodic Climb Height (m)", fontsize=20)
    plt.xlabel("")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()

if __name__ == "__main__":
    # --- 独立运行配置 ---
    N_EPISODES = config.N_EVAL_EPISODES
    POLAR_BASE = config.POLAR_BASE
    Q_TABLE_PATH = config.SAVE_PATH

    if not os.path.exists(Q_TABLE_PATH):
        print(f"未找到 Q 表: {Q_TABLE_PATH}")
    else:
        with open(Q_TABLE_PATH, "rb") as f:
            q_table_eval = pickle.load(f)

        # 使用 config 中的自然排序
        h5_files = sorted(glob.glob(os.path.join(config.WIND_DIR, 'snapshots_s*.h5')), key=config.natural_key)

        env_eval = GliderEnv(h5_file_path=h5_files, polar_file_base=POLAR_BASE)

        rnd_rewards, rnd_climbs = run_eval(env_eval, q_table_eval, N_EPISODES, "random")
        exp_rewards, exp_climbs = run_eval(env_eval, q_table_eval, N_EPISODES, "trained")
        
        print(f"--- Reward 统计 ---")
        print(f"随机策略: Mean reward={np.mean(rnd_rewards):.2f}")
        print(f"训练策略: Mean reward={np.mean(exp_rewards):.2f}")

        plot_climb_results(rnd_climbs, exp_climbs)
        env_eval.close()
