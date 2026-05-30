import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
import re

# ================= 配置 =================
MODEL_DIR = "practice/discrete swimmer"
SAVE_DIR = "practice/discrete swimmer/policy_plots"

def draw_single_policy(q_table_path, episode_num):
    # 加载 Q-Table (shape: 2, 2, 2, 2, 4)
    q_table = np.load(q_table_path)
    
    # 动作映射: (dx, dy, symbol, color)
    actions = {
        0: (0, 0.35, "↑", "blue"),    # Up
        1: (0.35, 0, "→", "green"),   # Right
        2: (0, -0.35, "↓", "red"),    # Down
        3: (-0.35, 0, "←", "orange")  # Left
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # 定义九宫格映射
    grid_cells = [
        (0, 2, [1, 0, 1, 0]), # NW
        (1, 2, [1, 0, 0, 0]), # N
        (2, 2, [1, 0, 0, 1]), # NE
        (0, 1, [0, 0, 1, 0]), # W
        (1, 1, None),         # Center (Agent)
        (2, 1, [0, 0, 0, 1]), # E
        (0, 0, [0, 1, 1, 0]), # SW
        (1, 0, [0, 1, 0, 0]), # S
        (2, 0, [0, 1, 0, 1]), # SE
    ]

    for gx, gy, state_vec in grid_cells:
        # 画格子
        rect = patches.Rectangle((gx-0.45, gy-0.45), 0.9, 0.9, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        if state_vec is None:
            # 中心格 Agent - 增大字体
            ax.text(gx, gy, "Agent", ha='center', va='center', fontsize=24, fontweight='bold')
            continue

        state = tuple(state_vec)
        q_values = q_table[state]
        
        best_action = np.argmax(q_values)
        adx, ady, label, color = actions[best_action]
        
        # 在格子内部画箭头 - 增加粗细 (width 参数)
        arrow_params = {'head_width': 0.2, 'head_length': 0.18, 'fc': color, 'ec': color, 'width': 0.06}
        if best_action == 0: # Up
            ax.arrow(gx, gy-0.2, 0, 0.3, **arrow_params)
        elif best_action == 1: # Right
            ax.arrow(gx-0.2, gy, 0.3, 0, **arrow_params)
        elif best_action == 2: # Down
            ax.arrow(gx, gy+0.2, 0, -0.3, **arrow_params)
        elif best_action == 3: # Left
            ax.arrow(gx+0.2, gy, -0.3, 0, **arrow_params)


    plt.title(f"Policy - Episode {episode_num}", fontsize=24, pad=20)
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    save_path = os.path.join(SAVE_DIR, f"policy_E{episode_num}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"已生成并保存: {save_path}")

def visualize_all():
    # 查找所有 q_table_EXXX.npy 文件
    files = glob.glob(os.path.join(MODEL_DIR, "q_table_E*.npy"))
    
    # 提取集数并排序
    policy_files = []
    for f in files:
        match = re.search(r'q_table_E(\d+)\.npy', f)
        if match:
            episode = int(match.group(1))
            policy_files.append((episode, f))
    
    # 按集数升序排列
    policy_files.sort()

    if not policy_files:
        print("未找到符合命名规则的 Q-Table 文件 (q_table_EXXX.npy)")
        return

    print(f"共找到 {len(policy_files)} 个策略文件，开始批量生成可视化...")
    for episode, path in policy_files:
        draw_single_policy(path, episode)
    
    print(f"\n批量可视化完成！所有图片保存在: {SAVE_DIR}")

if __name__ == "__main__":
    visualize_all()
