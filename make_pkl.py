import numpy as np
import pickle
import os
from glider_discrete_simp import GliderEnv

# Symbol key:
# 2: TriUp (Red Upward Triangle) - Action to increase bank
# 0: TriDn (Blue Downward Triangle) - Action to decrease bank
# 1: Sq (Brown Square) - Action to stay/hold bank

# Policy 1 (Low u_rms) - Transcribed from trainresult/policy1.png
# Rows are Bank Angle Index (0 to 6, corresponding to -15° to 15°)
# Columns are combinations of (acc_idx, dw_idx) as:
# Col 0-2: dw_idx=0 (Left strong) for acc_idx=0, 1, 2
# Col 3-5: dw_idx=1 (Balanced)    for acc_idx=0, 1, 2
# Col 6-8: dw_idx=2 (Right strong) for acc_idx=0, 1, 2
grid_low = [
    [1, 1, 2, 1, 2, 2, 0, 2, 2], # -15° (Index 0)
    [0, 2, 2, 0, 0, 2, 0, 1, 2], # -10° (Index 1)
    [2, 2, 2, 0, 0, 2, 0, 0, 1], # -5°  (Index 2)
    [2, 2, 2, 2, 2, 2, 0, 0, 0], # 0°   (Index 3)
    [2, 2, 1, 2, 2, 0, 0, 0, 0], # 5°   (Index 4)
    [2, 1, 0, 2, 2, 0, 2, 0, 0], # 10°  (Index 5)
    [2, 0, 0, 2, 0, 0, 1, 1, 0], # 15°  (Index 6)
]

# Policy 2 (High u_rms) - Transcribed from trainresult/policy2.png
grid_high = [
    [2, 1, 2, 2, 2, 2, 0, 2, 2], # -15° (Index 0)
    [2, 1, 2, 2, 2, 2, 0, 1, 2], # -10° (Index 1)
    [2, 2, 2, 0, 2, 2, 0, 0, 1], # -5°  (Index 2)
    [2, 2, 2, 0, 0, 0, 0, 0, 0], # 0°   (Index 3)
    [2, 2, 1, 2, 0, 0, 0, 0, 0], # 5°   (Index 4)
    [2, 1, 0, 0, 0, 0, 0, 1, 0], # 10°  (Index 5)
    [2, 0, 0, 0, 0, 0, 0, 1, 0], # 15°  (Index 6)
]

def generate_q_table(grid):
    # 初始化全零 Q-table (状态: [bank, acc, dw], 动作: 3)
    q_table = np.zeros((GliderEnv.BANK_BINS, 3, 3, 3))
    
    for b_idx in range(GliderEnv.BANK_BINS):
        for dw_idx in range(3):
            for acc_idx in range(3):
                # Columns mapping:
                # dw_idx=0 -> col 0, 1, 2
                # dw_idx=1 -> col 3, 4, 5
                # dw_idx=2 -> col 6, 7, 8
                col_idx = dw_idx * 3 + acc_idx
                best_action = grid[b_idx][col_idx]
                
                # 赋予该最优动作一个基础 Q 值
                q_table[b_idx, acc_idx, dw_idx, best_action] = 10.0
                
    return q_table

# 保存文件
Q_TABLE_DIR = "q_table"
os.makedirs(Q_TABLE_DIR, exist_ok=True)

for name, grid in [("q_table_low.pkl", grid_low), ("q_table_high.pkl", grid_high)]:
    q_table = generate_q_table(grid)
    save_path = os.path.join(Q_TABLE_DIR, name)
    with open(save_path, "wb") as f:
        pickle.dump(q_table, f)
    print(f"Policy '{name}' saved to {save_path}, shape: {q_table.shape}")
