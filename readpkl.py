import pickle
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from glider_discrete_simp import GliderEnv
import config

# --- 绘图配置 (加大字体) ---
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 14
})

def plot_q_table_policy(q_table_path, save_dir="trainresult"):
    # 1. 加载 Q-table
    try:
        with open(q_table_path, "rb") as f:
            q_table = pickle.load(f)
    except Exception as e:
        print(f"加载失败 {q_table_path}: {e}")
        return

    # 动态获取维度 (aoa, bank, acc, dw, action)
    n_aoa, n_bank, n_acc, n_dw, n_act = q_table.shape
    print(f"正在处理: {os.path.basename(q_table_path)} | Shape: {q_table.shape}")

    # 2. 选择要可视化的 AoA (中间索引)
    aoa_idx = 1
    aoa_deg = GliderEnv.AOA_MIN_DEG + aoa_idx * GliderEnv.AOA_STEP_DEG

    # 创建画布
    fig, axes = plt.subplots(n_bank, 1, figsize=(7, n_bank * 1.5), sharex=True)
    if n_bank == 1:
        axes = [axes]

    # 状态标签
    symbols = GliderEnv.OBS_WIND_SYMBOLS
    action_labels = GliderEnv.ACTION_LABELS

    # 3. 循环绘图
    for b_idx in range(n_bank):
        ax = axes[n_bank - 1 - b_idx]
        
        # 假设 bank 对称分布
        bank_center = n_bank // 2
        bank_deg = (b_idx - bank_center) * GliderEnv.BANK_STEP_DEG
        
        obs_labels = []
        best_actions = []
        
        for dw_idx in range(n_dw):
            for acc_idx in range(n_acc):
                obs_labels.append(f"{symbols[acc_idx]}|{symbols[dw_idx]}")
                best_actions.append(np.argmax(q_table[aoa_idx, b_idx, acc_idx, dw_idx]))

        for i, action in enumerate(best_actions):
            full_label = action_labels.get(action, "?")
            
            if full_label == "?":
                ax.text(i, 0, "?", ha='center', va='center', fontsize=20)
                continue
                
            # 1. 确定滚转 (Bank) 对应的形状和边缘颜色
            marker_shape = 'o'
            edge_color = 'green'
            if "B+" in full_label:
                marker_shape = '^'
                edge_color = 'red'
            elif "B-" in full_label:
                marker_shape = 'v'
                edge_color = 'blue'
                
            # 2. 确定迎角 (AoA) 对应的填充颜色
            face_color = 'none'  # 默认空心 (A0)
            if "A+" in full_label:
                face_color = 'yellow'  # 黄色增加
            elif "A-" in full_label:
                face_color = 'black'   # 黑色减少
                
            # 3. 绘制组合图标
            ax.plot(i, 0, marker=marker_shape, markersize=24, 
                    markerfacecolor=face_color, markeredgecolor=edge_color, 
                    markeredgewidth=3, linestyle='None')

        ax.set_yticks([])
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(-0.5, (n_dw * n_acc) - 0.5)
        ax.set_ylabel(f"Bank {bank_deg:+.0f}°", rotation=0, labelpad=60, va='center')
        
        for spine in ["top", "left", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_alpha(0.3)

    plt.suptitle(f"Policy: {os.path.basename(q_table_path)}\n(AoA = {aoa_deg:.1f}°)", fontsize=22)
    axes[-1].set_xticks(range(n_dw * n_acc))
    axes[-1].set_xticklabels(obs_labels)
    axes[-1].set_xlabel(r"Wind State ($a_z|\tau$)", labelpad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    file_stem = os.path.splitext(os.path.basename(q_table_path))[0]
    save_path = os.path.join(save_dir, f"policy_{file_stem}.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"已保存: {save_path}")

if __name__ == "__main__":
    # 获取所有 E_xxxx 格式的 pkl 文件
    pkl_files = sorted(glob.glob(os.path.join(config.Q_TABLE_DIR, "q_table_E_*.pkl")), key=config.natural_key)
    
    if pkl_files:
        print(f"找到 {len(pkl_files)} 个匹配的 Q-table 文件，开始批量可视化...")
        for f in pkl_files:
            plot_q_table_policy(f)
        print("所有文件处理完毕。")
    else:
        print(f"在 {config.Q_TABLE_DIR} 中未找到符合 q_table_E_*.pkl 模式的文件。")