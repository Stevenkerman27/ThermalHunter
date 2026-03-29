import numpy as np
import pickle
import os
from glider_discrete_simp import GliderEnv

# 初始化全零 Q-table (状态: [bank, acc, dw], 动作: 3)
# 状态 0: bank (0-6), 状态 1: acc (0-2), 状态 2: dw (0-2)
q_table = np.zeros((GliderEnv.BANK_BINS, 3, 3, 3))

for b_idx in range(GliderEnv.BANK_BINS):
    for acc_idx in range(3):     # 垂直加速度索引: 0(负), 1(零), 2(正)
        for dw_idx in range(3):  # 翼尖风速差索引: 0(左强), 1(平衡), 2(右强)
            
            # --- 核心逻辑: 滚转策略 ---
            # 如果左侧强 (dw_idx=0), 应该向右滚转 (增加 bank, action=2)
            # 如果右侧强 (dw_idx=2), 应该向左滚转 (减小 bank, action=0)
            if dw_idx == 0:
                best_action = 2
            elif dw_idx == 2:
                best_action = 0
            else:
                # 平衡状态下，如果不在 0 滚转 (b_idx=3)，则尝试回归 0
                if b_idx < 3:
                    best_action = 2
                elif b_idx > 3:
                    best_action = 0
                else:
                    best_action = 1
            
            # 赋予该最优动作一个基础 Q 值
            q_table[b_idx, acc_idx, dw_idx, best_action] = 10.0

# 保存文件
Q_TABLE_DIR = "q_table"
os.makedirs(Q_TABLE_DIR, exist_ok=True)
save_name = os.path.join(Q_TABLE_DIR, "q_table_ideal.pkl")
with open(save_name, "wb") as f:
    pickle.dump(q_table, f)

print(f"理想策略已成功写入 {save_name}, shape: {q_table.shape}")