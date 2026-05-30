import torch
import torch.nn as nn
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import config
from train_dqn import QNetwork, normalize_state

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path, state_dim, action_dim):
    model = QNetwork(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def generate_decision_map(model, sensor_stats, aoa_idx, bank_idx):
    # Ranges from stats
    w_accel_mean = sensor_stats["w_accel"]["mean"]
    w_accel_std = sensor_stats["w_accel"]["std"]
    delta_w_mean = sensor_stats["delta_w"]["mean"]
    delta_w_std = sensor_stats["delta_w"]["std"]
    
    # 100x100 grid covering +/- 3 sigma
    n_points = 100
    w_accel_vals = np.linspace(w_accel_mean - 3*w_accel_std, w_accel_mean + 3*w_accel_std, n_points)
    delta_w_vals = np.linspace(delta_w_mean - 3*delta_w_std, delta_w_mean + 3*delta_w_std, n_points)
    
    W, D = np.meshgrid(w_accel_vals, delta_w_vals)
    action_grid = np.zeros_like(W, dtype=int)
    
    for i in range(n_points):
        for j in range(n_points):
            raw_state = np.array([aoa_idx, bank_idx, W[i, j], D[i, j]])
            norm_state = normalize_state(raw_state)
            state_t = torch.FloatTensor(norm_state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = model(state_t).squeeze()
                best_action = q_values.argmax().item()
                
                # 迟滞 (Hysteresis) 逻辑
                q_range = q_values.max() - q_values.min()
                threshold = max(config.DQN_ACTION_MARGIN_MIN, config.DQN_ACTION_MARGIN_K * q_range)
                
                # 如果最优动作不是中性动作(4)，检查其领先优势是否超过阈值
                if best_action != 4 and q_values[best_action] < q_values[4] + threshold:
                    final_action = 4
                else:
                    final_action = best_action
            
            action_grid[i, j] = final_action
                
    return W, D, action_grid

def main():
    # --- Configuration ---
    target_aoa_idx = 3
    target_bank_idx = 2
    
    aoa_deg = config.AOA_MIN_DEG + target_aoa_idx * config.AOA_STEP_DEG
    bank_deg = config.BANK_MIN_DEG + target_bank_idx * config.BANK_STEP_DEG

    stats_path = os.path.join(config.BASE_DIR, "sensor_stats.json")
    with open(stats_path, "r") as f:
        sensor_stats = json.load(f)
        
    model = load_model(config.DQN_SAVE_PATH, 4, 9)
    
    W, D, action_grid = generate_decision_map(model, sensor_stats, target_aoa_idx, target_bank_idx)
    
    # Detailed Action Distribution
    unique, counts = np.unique(action_grid, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Full Action Distribution:")
    for a_idx in range(9):
        aoa_delta = (a_idx // 3) - 1
        bank_delta = (a_idx % 3) - 1
        count = dist.get(a_idx, 0)
        label = f"A{aoa_delta:+}B{bank_delta:+}"
        if a_idx == 4: label = "HOLD"
        print(f"  Action {a_idx} ({label}): {count}")
    
    # 9 Distinct Colors
    # Layout (3x3):
    # 0:A-B- 1:A-B0 2:A-B+
    # 3:A0B- 4:A0B0 5:A0B+
    # 6:A+B- 7:A+B0 8:A+B+
    colors = [
        '#ff9999', '#ffcc99', '#ffff99', # A- (Reds/Oranges)
        '#99ff99', '#f0f0f0', '#99ffff', # A0 (Green/Grey/Cyan)
        '#9999ff', '#cc99ff', '#ff99ff'  # A+ (Blues/Purples)
    ]
    cmap = ListedColormap(colors)
    
    plt.figure(figsize=(11, 8))
    # Using vmin/vmax to ensure all 9 colors are mapped correctly even if some actions are missing
    plt.pcolormesh(W, D, action_grid, cmap=cmap, shading='auto', alpha=0.9, vmin=0, vmax=8)
    
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)
    
    plt.xlabel(r'Vertical Acceleration $a_z$ (m/s$^2$)')
    plt.ylabel(r'Wing-tip Difference $\Delta w$ (m/s)')
    plt.title(f'DQN Full Policy Map with Hysteresis (Bank={bank_deg:+.0f}°, AoA={aoa_deg:.1f}°)\n'
              r'Hysteresis Margin={:.1f}% | 9 Discrete Actions Shown'.format(config.DQN_ACTION_MARGIN_K*100))
    
    # Add legend for all 9 actions
    from matplotlib.patches import Patch
    legend_elements = []
    for a_idx in range(9):
        aoa_delta = (a_idx // 3) - 1
        bank_delta = (a_idx % 3) - 1
        label = f"Action {a_idx}: AoA{aoa_delta:+} Bank{bank_delta:+}"
        if a_idx == 4: label = "Action 4: HOLD (Neutral)"
        legend_elements.append(Patch(facecolor=colors[a_idx], label=label))
    
    # Place legend outside
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(config.TRAIN_RESULT_DIR, "dqn_neutral_slice.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
