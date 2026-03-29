import pickle
import numpy as np
import matplotlib.pyplot as plt


q_table = pickle.load(open("q_table_ideal.pkl", "rb"))

# 状态标签排序 (dw 为外循环，acc 为内循环，匹配图示顺序)
symbols = ["-", "0", "+"]
obs_labels = []
best_actions = []

for dw in range(3):
    for acc in range(3):
        obs_labels.append(f"{symbols[acc]}|{symbols[dw]}")
        best_actions.append(np.argmax(q_table[acc, dw]))

# 3. 箭头映射函数 (基于 glider_discrete_simp.py 的动作定义)
def get_arrow_style(action):
    # 标记映射：(LaTeX箭头, 颜色)
    # 0-2: Low AoA (Blue), 3-5: Mid AoA (Orange), 6-8: High AoA (Red)
    mapping = {
        0: (r"$\swarrow$", 'blue'),   1: (r"$\downarrow$", 'blue'), 2: (r"$\searrow$", 'blue'),
        3: (r"$\leftarrow$", 'orange'), 4: (r"$\bullet$", 'green'),    5: (r"$\rightarrow$", 'orange'),
        6: (r"$\nwarrow$", 'red'),    7: (r"$\uparrow$", 'red'),    8: (r"$\nearrow$", 'red')
    }
    return mapping.get(action)

# 4. 绘图 (单行排列)
fig, ax = plt.subplots(figsize=(10, 2))

for i, action in enumerate(best_actions):
    marker, color = get_arrow_style(action)
    # 在 y=0 处绘制散点，使用 LaTeX 箭头作为 marker
    ax.scatter(i, 0, marker=marker, s=1000, color=color)

# 坐标轴美化
ax.set_xticks(range(9))
ax.set_xticklabels(obs_labels, fontsize=12)
ax.set_yticks([]) 
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(-0.5, 8.5)
ax.set_xlabel(r"State Combination ($a_z | \tau$)", fontsize=12, labelpad=10)

# 隐藏多余边框
for spine in ["top", "left", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_position(("data", -0.2))

plt.tight_layout()
plt.show()