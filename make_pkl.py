import numpy as np
import pickle
import os
import config
import matplotlib.pyplot as plt
from glider_discrete_simp import GliderEnv

# Symbol key:
# 2: TriUp (Red Upward Triangle) - Action to increase bank (B+)
# 0: TriDn (Blue Downward Triangle) - Action to decrease bank (B-)
# 1: Sq/Circ (Brown Square / Green Circle) - Action to stay/hold bank (B0)

# In the 9-action environment:
# Action 3: A0B- (Decrease bank, neutral AoA)
# Action 4: A0B0 (Stay, neutral AoA)
# Action 5: A0B+ (Increase bank, neutral AoA)
ACTION_MAP = {
    0: 3, # B- -> A0B-
    1: 4, # B0 -> A0B0
    2: 5  # B+ -> A0B+
}

# 7-bin policies based on provided images
# Rows are Bank Angle Index (0 to 6, corresponding to -15° to +15°)
grid_oldpolicy = [
    [2, 2, 2, 0, 1, 2, 0, 1, 2], # -15°
    [2, 2, 2, 2, 2, 2, 0, 0, 1], # -10°
    [2, 2, 2, 2, 0, 2, 0, 0, 0], # -5°
    [2, 2, 2, 2, 0, 2, 0, 0, 0], #   0°
    [2, 2, 2, 2, 2, 2, 0, 0, 0], # +5°
    [2, 2, 2, 2, 2, 0, 0, 0, 0], # +10°
    [1, 0, 0, 2, 0, 0, 0, 0, 0], # +15°
]

grid_low = [
    [1, 1, 2, 1, 2, 2, 0, 2, 2], # -15°
    [0, 2, 2, 0, 0, 2, 0, 1, 2], # -10°
    [2, 2, 2, 0, 0, 2, 0, 0, 1], # -5°
    [2, 2, 2, 2, 2, 2, 0, 0, 0], #   0°
    [2, 2, 1, 2, 2, 0, 0, 0, 0], # +5°
    [2, 1, 0, 2, 2, 0, 2, 0, 0], # +10°
    [2, 0, 0, 2, 0, 0, 1, 1, 0], # +15°
]

grid_high = [
    [2, 1, 2, 2, 2, 2, 0, 2, 2], # -15°
    [2, 1, 2, 2, 2, 2, 0, 1, 2], # -10°
    [2, 2, 2, 0, 2, 2, 0, 0, 1], # -5°
    [2, 2, 2, 0, 0, 0, 0, 0, 0], #   0°
    [2, 2, 1, 2, 0, 0, 0, 0, 0], # +5°
    [2, 1, 0, 0, 0, 0, 0, 1, 0], # +10°
    [2, 0, 0, 0, 0, 0, 0, 1, 0], # +15°
]

def generate_q_table(grid):
    # 初始化 Q-table (状态: [aoa, bank, acc, dw], 动作: 9)
    aoa_bins = GliderEnv.AOA_BINS
    bank_bins = 7 # 强制 7-bin
    q_table = np.zeros((aoa_bins, bank_bins, 3, 3, 9))
    
    for b_idx in range(bank_bins):
        for dw_idx in range(3):
            for acc_idx in range(3):
                col_idx = dw_idx * 3 + acc_idx
                bank_action_val = grid[b_idx][col_idx]
                
                # 映射到 9 动作空间中的 A0 行 (Action 3, 4, 5)
                best_action = ACTION_MAP.get(bank_action_val, 4)
                
                # 赋予该最优动作一个基础 Q 值
                for a_idx in range(aoa_bins):
                    q_table[a_idx, b_idx, acc_idx, dw_idx, best_action] = 10.0
                
    return q_table

# 保存文件
Q_TABLE_DIR = "q_table"
os.makedirs(Q_TABLE_DIR, exist_ok=True)

policies_to_gen = [
    ("q_table_oldpolicy.pkl", grid_oldpolicy),
    ("q_table_low.pkl", grid_low),
    ("q_table_high.pkl", grid_high)
]

def plot_policy(q_table, policy_name):
    AOA_TO_PLOT = 1
    aoa_deg = GliderEnv.AOA_MIN_DEG + AOA_TO_PLOT * GliderEnv.AOA_STEP_DEG

    BANK_BINS = 7
    BANK_MIN_DEG = -15
    BANK_STEP_DEG = 5

    fig, axes = plt.subplots(BANK_BINS, 1, figsize=(6.5, BANK_BINS), sharex=True)
    if BANK_BINS == 1:
        axes = [axes]

    # 状态标签
    symbols = GliderEnv.OBS_WIND_SYMBOLS # ["-", "0", "+"]
    action_labels = GliderEnv.ACTION_LABELS

    # 3. 循环绘图
    for b_idx in range(BANK_BINS):
        # 反转索引，使正角度在最上面，负角度在最下面
        ax = axes[BANK_BINS - 1 - b_idx]
        bank_deg = BANK_MIN_DEG + b_idx * BANK_STEP_DEG
        
        obs_labels = []
        best_actions = []
        
        # 遍历风场状态 (dw 为外循环，acc 为内循环)
        for dw_idx in range(3):
            for acc_idx in range(3):
                obs_labels.append(f"{symbols[acc_idx]}|{symbols[dw_idx]}")
                # Q-table 索引顺序: [aoa, bank, acc, dw, action]
                best_actions.append(np.argmax(q_table[AOA_TO_PLOT, b_idx, acc_idx, dw_idx]))

        # 绘制动作文本/标识
        for i, action in enumerate(best_actions):
            full_label = action_labels.get(action, "?")
            
            # 根据动作分量决定符号和颜色 (仅基于 Bank 动作)
            if "B0" in full_label:
                ax.plot(i, 0, marker='o', markersize=20, markerfacecolor='green', markeredgecolor='green', linestyle='None')
            elif "B+" in full_label:
                ax.plot(i, 0, marker='^', markersize=24, markerfacecolor='none', markeredgecolor='red', markeredgewidth=3, linestyle='None')
            elif "B-" in full_label:
                ax.plot(i, 0, marker='v', markersize=24, markerfacecolor='none', markeredgecolor='blue', markeredgewidth=3, linestyle='None')
            else:
                ax.text(i, 0, "?", ha='center', va='center', fontsize=24)

        # 子图装饰
        ax.set_yticks([])
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylabel(f"Bank {bank_deg:+.0f}°", rotation=0, labelpad=45, va='center', fontsize=18)
        
        # 隐藏边框
        for spine in ["top", "left", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_alpha(0.3)

    plt.suptitle("Q-Table Visualization", fontsize=18)
    # 设置最底部的 X 轴标签
    axes[-1].set_xticks(range(9))
    axes[-1].set_xticklabels(obs_labels, fontsize=20)
    axes[-1].set_xlabel(r"Wind State ($a_z|\tau$)", fontsize=20, labelpad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("q_table", exist_ok=True)
    
    stem = policy_name.replace("q_table_", "").replace(".pkl", "")
    save_path = f"q_table/{stem}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Policy plot saved to {save_path}")

for name, grid in policies_to_gen:
    q_table = generate_q_table(grid)
    save_path = os.path.join(Q_TABLE_DIR, name)
    with open(save_path, "wb") as f:
        pickle.dump(q_table, f)
    print(f"Policy '{name}' saved to {save_path}, shape: {q_table.shape}")
    plot_policy(q_table, name)