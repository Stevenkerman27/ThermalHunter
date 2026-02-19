import numpy as np
import pickle

# 初始化全零 Q-table (3x3x3 状态, 9 动作)
q_table = np.zeros((3, 3, 3, 9))

for v in range(3):           # 速度索引
    for acc in range(3):     # 加速度索引
        for dw in range(3):  # 翼尖差索引
            
            # --- 1. 确定迎角逻辑 (AoA) ---
            if v == 0:        # 规则：速度低则恒定减小迎角（俯冲增速）
                target_aoa = "dec"
            elif v == 2:      # 规则：速度高则增加迎角（拉起减速）
                target_aoa = "inc"
            else:             # 规则：常速下根据垂直加速度判断
                if acc == 2:    # 正加速度 -> 增迎角
                    target_aoa = "inc"
                elif acc == 0:  # 负加速度 -> 减迎角
                    target_aoa = "dec"
                else:           # 无加速度 -> 保持
                    target_aoa = "keep"

            # --- 2. 确定滚转逻辑 (Bank) ---
            # 规则：向有翼尖速度差的一边滚转 (delta_w = w_right - w_left)
            if dw == 2:       # 右翼升力大 (dw > 0.08) -> 向右滚转
                target_bank = "right"
            elif dw == 0:     # 左翼升力大 (dw < -0.08) -> 向左滚转
                target_bank = "left"
            else:             # 差值小 -> 不滚转
                target_bank = "none"

            # --- 3. 映射到具体动作索引 ---
            # 根据 target_aoa 和 target_bank 组合寻找 action ID
            action_map = {
                ("inc", "right"): 0, ("inc", "none"): 1, ("inc", "left"): 2,
                ("keep", "right"): 3, ("keep", "none"): 4, ("keep", "left"): 5,
                ("dec", "right"): 6, ("dec", "none"): 7, ("dec", "left"): 8
            }
            
            best_action = action_map[(target_aoa, target_bank)]
            
            # 赋予该动作一个较高的初始 Q 值（例如 100）
            q_table[v, acc, dw, best_action] = 10

# 保存文件
with open("q_table_ideal.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("策略已成功写入 q_table_v0.pkl")