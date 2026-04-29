import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from glider_discrete_simp import GliderEnv

# 1. 加载 Q-table
# 注意：现在 Q-table 的形状应为 (AOA_BINS, BANK_BINS, 3, 3, 9)
Q_TABLE_DIR = "q_table"
Q_TABLE_PATH = os.path.join(Q_TABLE_DIR, "q_table_v0.pkl")

q_table = pickle.load(open(Q_TABLE_PATH, "rb"))
print(f"成功加载 Q-table: {Q_TABLE_PATH}, shape: {q_table.shape}")

# 2. 绘图配置
# 我们将针对中间的 AoA 索引，为 7 个不同的 Bank Angle 分别绘制一个 3x3 的风场状态矩阵
AOA_TO_PLOT = GliderEnv.AOA_BINS // 2
aoa_deg = GliderEnv.AOA_MIN_DEG + AOA_TO_PLOT * GliderEnv.AOA_STEP_DEG

fig, axes = plt.subplots(GliderEnv.BANK_BINS, 1, figsize=(10, GliderEnv.BANK_BINS), sharex=True)
if GliderEnv.BANK_BINS == 1:
    axes = [axes]

# 状态标签
symbols = GliderEnv.OBS_WIND_SYMBOLS # ["-", "0", "+"]
# action_mapping: {0: A-B-, 1: A-B0, 2: A-B+, 3: A0B-, 4: A0B0, 5: A0B+, 6: A+B-, 7: A+B0, 8: A+B+}
action_labels = GliderEnv.ACTION_LABELS

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
            # Q-table 索引顺序: [aoa, bank, acc, dw, action]
            best_actions.append(np.argmax(q_table[AOA_TO_PLOT, b_idx, acc_idx, dw_idx]))

    # 绘制动作文本/标识
    for i, action in enumerate(best_actions):
        label = action_labels.get(action, "?")
        # 根据动作分量决定颜色 (仅作为示例，这里逻辑可以根据需要调整)
        # 4 是 A0B0 (保持)，颜色设为绿色
        color = 'green' if action == 4 else 'red'
        if "A-" in label or "B-" in label: color = 'blue'
        
        ax.text(i, 0, label, ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    # 子图装饰
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylabel(f"Bank {bank_deg:+.0f}°", rotation=0, labelpad=40, va='center', fontsize=12)
    
    # 隐藏边框
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_alpha(0.3)

plt.suptitle(f"Policy Map (AoA = {aoa_deg:.1f}°)\nLabels: A(AoA) B(Bank), +/- (Inc/Dec), 0 (Keep)", fontsize=14)
# 设置最底部的 X 轴标签
axes[-1].set_xticks(range(9))
axes[-1].set_xticklabels(obs_labels, fontsize=12)
axes[-1].set_xlabel(r"State ($a_z | \tau$)", fontsize=12, labelpad=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("trainresult/policy.png", dpi=300)
plt.show()