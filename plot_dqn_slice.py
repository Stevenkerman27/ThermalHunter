import torch
import torch.nn as nn
import numpy as np
import os
import json
import glob
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
    
    # Vectorized state generation
    w_flat = W.flatten()
    d_flat = D.flatten()
    n = len(w_flat)
    
    aoa_vals = np.full(n, aoa_idx)
    bank_vals = np.full(n, bank_idx)
    
    # raw_states shape (N, 4)
    raw_states = np.stack([aoa_vals, bank_vals, w_flat, d_flat], axis=1)
    
    # Vectorized normalization (matching train_dqn.normalize_state logic)
    norm_states = raw_states.copy().astype(np.float32)
    norm_states[:, 0] = (norm_states[:, 0] - (config.AOA_BINS / 2)) / (config.AOA_BINS / 2)
    norm_states[:, 1] = (norm_states[:, 1] - (config.BANK_BINS / 2)) / (config.BANK_BINS / 2)
    norm_states[:, 2] = (norm_states[:, 2] - w_accel_mean) / w_accel_std
    norm_states[:, 3] = (norm_states[:, 3] - delta_w_mean) / delta_w_std
    
    states_t = torch.FloatTensor(norm_states).to(device)
    
    with torch.no_grad():
        q_values = model(states_t)
        final_actions = q_values.argmax(dim=1).cpu().numpy()
            
    action_grid = final_actions.reshape(W.shape)
    return W, D, action_grid

def main():
    # --- Plotting Font Standards ---
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 22,
        'axes.titlesize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20
    })

    stats_path = os.path.join(config.BASE_DIR, "sensor_stats.json")
    with open(stats_path, "r") as f:
        sensor_stats = json.load(f)
        
    # Find the final model
    model_files = glob.glob(os.path.join(config.Q_TABLE_DIR, "*.cleanrl_model"))
    if os.path.exists(config.DQN_SAVE_PATH):
        model_files.append(config.DQN_SAVE_PATH)
    # Deduplicate and sort
    model_files = sorted(list(set(model_files)), key=config.natural_key)
    
    if not model_files:
        print("No model files found in q_table directory.")
        return

    # Use the final one
    model_path = model_files[-1]
    model_name = os.path.basename(model_path).replace(".cleanrl_model", "").replace(".pth", "")
    print(f"Generating 3x3 policy grid for final model: {model_name}")
    
    model = load_model(model_path, 4, 9)
    
    # New Colorblind-friendly colors (AoA Delta)
    # AoA -1: Vermillion, AoA 0: Light Grey, AoA +1: Blue
    aoa_colors = {
        -1: '#D55E00', 
         0: '#CCCCCC', 
         1: '#0072B2'
    }
    
    # Pattern mapping (Bank Delta)
    # Bank -1: \\, Bank 0: None, Bank +1: //
    bank_hatches = {
        -1: '\\\\',
         0: '',
         1: '////'
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    
    for r in range(3): # AoA index row
        for c in range(3): # Bank index column
            ax = axes[r, c]
            aoa_idx = r
            bank_idx = c
            
            W, D, action_grid = generate_decision_map(model, sensor_stats, aoa_idx, bank_idx)
            
            ax.pcolormesh(W, D, action_grid, cmap=cmap, shading='auto', alpha=0.9, vmin=0, vmax=8)
            
            ax.axhline(0, color='black', linestyle='--', alpha=0.2)
            ax.axvline(0, color='black', linestyle='--', alpha=0.2)
            
            # Ensure 1:1 aspect ratio for the plot box
            ax.set_box_aspect(1)
            
            if r == 2: ax.set_xlabel(r'$a_z$ (m/s$^2$)')
            if c == 0: ax.set_ylabel(r'$\tau$ (m/s)')

    # Add shared legend
    from matplotlib.patches import Patch
    legend_elements = []
    for a_idx in range(9):
        aoa_delta = (a_idx // 3) - 1
        bank_delta = (a_idx % 3) - 1
        label = f"AoA{aoa_delta:+} Bank{bank_delta:+}"
        if a_idx == 4: label = "HOLD"
        legend_elements.append(Patch(facecolor=colors[a_idx], label=label))
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.005),
               ncol=3, frameon=True, handletextpad=0.5, columnspacing=1.0)
    
    # Adjust spacing between subplots (wspace/hspace) and figure margins
    # Seamless grid
    plt.tight_layout(rect=[0, 0.11, 1, 1])
    plt.subplots_adjust(wspace=0.0, hspace=0.03)
    
    output_path = os.path.join(config.TRAIN_RESULT_DIR, f"dqn_policy_grid_{model_name}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grid plot saved to {output_path}")

if __name__ == "__main__":
    main()
