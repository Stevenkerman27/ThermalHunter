import numpy as np
import pickle

# 初始化全零 Q-table (状态: [acc, dw], 动作: 9)
# 状态 0: acc (加速度), 状态 1: dw (翼尖差)
q_table = np.zeros((3, 3, 9))

for acc in range(3):     # 垂直加速度索引: 0(负), 1(零), 2(正)
    for dw in range(3):  # 翼尖风速差索引: 0(左强), 1(平衡), 2(右强)
        
        # --- 1. 迎角逻辑 (AoA) ---
        # 0(负) -> 低迎角 (对应 Action 0,1,2)
        # 1(零) -> 中迎角 (对应 Action 3,4,5)
        # 2(正) -> 高迎角 (对应 Action 6,7,8)
        aoa_base = acc * 3 
        #aoa_base = (2 - acc) * 3
        # --- 2. 滚转逻辑 (Bank) ---
        # 0(左强) -> 左转 (Action +0)
        # 1(平衡) -> 直飞 (Action +1)
        # 2(右强) -> 右转 (Action +2)
        bank_offset = 2 - dw
        
        # --- 3. 组合动作索引 ---
        # 策略：action = acc * 3 + dw
        best_action = aoa_base + bank_offset
        
        # 赋予该最优动作一个基础 Q 值
        q_table[acc, dw, best_action] = 10.0

# 保存文件
save_name = "q_table_ideal.pkl"
with open(save_name, "wb") as f:
    pickle.dump(q_table, f)

print(f"理想策略已成功写入 {save_name}")