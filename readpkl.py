import pickle
import numpy as np

# 设置打印精度，防止数值过长导致排版错乱
np.set_printoptions(precision=3, suppress=True)

with open("q_table_v0.pkl", "rb") as f:
    q_table = pickle.load(f)

# --- 第一部分：打印原始 Q-table 数值 ---
print("### Full Q-Table Values (3x3x9) ###")
print("-" * 60)
for acc in range(3):
    for dw in range(3):
        values = q_table[acc, dw]
        print(f"State ({acc}, {dw}): {values}")
print("\n")

# --- 第二部分：打印最佳策略摘要 ---
print(f"{'State (acc, dw)':<18} | {'Best Action':<12} | {'Physical Maneuver'}")
print("-" * 60)

for acc in range(3):
    for dw in range(3):
        action = np.argmax(q_table[acc, dw])
        # 物理意义映射
        pitch = ["Low", "Mid", "High"][action // 3]
        roll = ["Left", "Straight", "Right"][action % 3]
        print(f"({acc}, {dw}){' ':<12} | {action:<12} | {pitch} + {roll}")