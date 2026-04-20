import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from glider_discrete_simp import GliderEnv

# 1. 加载 Q-table
# 注意：现在 Q-table 的形状应为 (BANK_BINS, 3, 3, 3) -> (Bank, Accel, Diff, Action)
Q_TABLE_DIR = "q_table"
#Q_TABLE_PATH = os.path.join(Q_TABLE_DIR, "q_table_high.pkl")
Q_TABLE_PATH = os.path.join(Q_TABLE_DIR, "q_table_v0.pkl")

q_table = pickle.load(open(Q_TABLE_PATH, "rb"))
print(f"成功加载 Q-table: {Q_TABLE_PATH}, shape: {q_table.shape}")

# 2. 绘图配置
# 我们将为 7 个不同的 Bank Angle 分别绘制一个 3x3 的风场状态矩阵
fig, axes = plt.subplots(GliderEnv.BANK_BINS, 1, figsize=(8, GliderEnv.BANK_BINS), sharex=True)
if GliderEnv.BANK_BINS == 1:
    axes = [axes]

# 状态标签
symbols = GliderEnv.OBS_WIND_SYMBOLS # ["-", "0", "+"]
action_mapping = GliderEnv.ACTION_LABELS # {0: bank-5, 1: bank+0, 2: bank+5}

# 3. 循环绘图
for b_idx in range(GliderEnv.BANK_BINS):
    # 反转索引，使 15° 在最上面，-15° 在最下面
    ax = axes[GliderEnv.BANK_BINS - 1 - b_idx]
    bank_deg = GliderEnv.BANK_MIN_DEG + b_idx * GliderEnv.BANK_STEP_DEG
    
    obs_labels = []
    best_actions = []
    
    # 遍历风场状态 (dw 为外循环，acc 为内循环)
    for dw_idx in range(3):
        for acc_idx in range(3):
            obs_labels.append(f"{symbols[acc_idx]}|{symbols[dw_idx]}")
            # Q-table 索引顺序: [bank, acc, dw, action]
            best_actions.append(np.argmax(q_table[b_idx, acc_idx, dw_idx]))

    # 绘制动作箭头
    for i, action in enumerate(best_actions):
        marker = action_mapping.get(action, "?")
        # 0: 蓝色 (减小), 1: 绿色 (保持), 2: 红色 (增加)
        color = 'blue' if action == 0 else ('green' if action == 1 else 'red')
        ax.scatter(i, 0, marker=marker, s=600, color=color)

    # 子图装饰
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylabel(f"Bank {bank_deg:+.0f}°", rotation=0, labelpad=40, va='center', fontsize=14)
    
    # 隐藏边框
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_alpha(0.3)

# 设置最底部的 X 轴标签
axes[-1].set_xticks(range(9))
axes[-1].set_xticklabels(obs_labels, fontsize=14)
axes[-1].set_xlabel(r"State ($a_z | \tau$)", fontsize=14, labelpad=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("trainresult/policy.png", dpi=300)
plt.show()