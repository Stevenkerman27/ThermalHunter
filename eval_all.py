import numpy as np
import gymnasium as gym
import torch
import os
import glob
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import config
from glider_discrete_simp import GliderEnv, RBWindField, GliderPhysics
from train_dqn import QNetwork, normalize_state

def get_tabular_obs(env, phy_state, aoa_idx, bank_idx, w_accel, delta_w, last_idx_az, last_idx_dw):
    """
    Replicates the Tabular observation mapping from GliderEnv.
    """
    # apply hysteresis
    idx_az = env._apply_hysteresis(w_accel, config.BINS_W_ACCEL, last_idx_az)
    idx_dw = env._apply_hysteresis(delta_w, config.BINS_DELTA_W, last_idx_dw)
    return np.array([aoa_idx, bank_idx, idx_az, idx_dw], dtype=np.int32), idx_az, idx_dw

class MultiGliderEvaluator:
    def __init__(self, h5_file_paths):
        # 1. Initialize Wind Field (Shared)
        self.wind_manager = RBWindField(h5_file_paths, domain_size=config.DOMAIN_SIZE, memory_mode=True)
        self.physics = GliderPhysics(config.POLAR_BASE)
        
        # 2. Precompute physics tables (Normal & Drag-Penalty)
        self._precompute_physics()
        
    def _precompute_physics(self):
        """Replicates GliderEnv._precompute_physics"""
        self.physics_table = np.zeros((config.AOA_BINS, config.BANK_BINS, 3), dtype=np.float32)
        self.physics_table_drag = np.zeros((config.AOA_BINS, config.BANK_BINS, 3), dtype=np.float32)
        
        for a_idx in range(config.AOA_BINS):
            aoa_rad = np.deg2rad(config.AOA_MIN_DEG + a_idx * config.AOA_STEP_DEG)
            for b_idx in range(config.BANK_BINS):
                bank_rad = np.deg2rad(config.BANK_MIN_DEG + b_idx * config.BANK_STEP_DEG)
                v_tas, gamma, dchi_dt = self.physics.get_steady_state(aoa_rad, bank_rad)
                self.physics_table[a_idx, b_idx] = [v_tas, gamma, dchi_dt]
                v_tas_d, gamma_d, dchi_dt_d = self.physics.get_steady_state(aoa_rad, bank_rad, drag_mult=config.CONTROL_DRAG_MULTIPLIER)
                self.physics_table_drag[a_idx, b_idx] = [v_tas_d, gamma_d, dchi_dt_d]

    def reset_episode(self, reset_time):
        self.wind_manager.reset(reset_time)
        
        # Initial state (Same for all 3 gliders)
        x, y = np.random.uniform(0.2, 0.8, size=2) * config.DOMAIN_SIZE[:2]
        z = np.random.uniform(0.2, 0.6) * config.DOMAIN_SIZE[2]
        init_dir = np.random.uniform(0, 2*np.pi)
        
        # [phy_state, aoa_idx, bank_idx, w_accel, delta_w, last_idx_az, last_idx_dw, done, h_start, ep_reward]
        self.gliders = []
        for _ in range(3): # Random, Tabular, DQN
            self.gliders.append({
                "phy_state": np.array([x, y, z, init_dir]),
                "aoa_idx": config.AOA_BINS // 2,
                "bank_idx": config.BANK_BINS // 2,
                "w_accel": 0.0,
                "delta_w": 0.0,
                "last_idx_az": None,
                "last_idx_dw": None,
                "done": False,
                "h_start": z,
                "ep_reward": 0.0,
                "h_final": z
            })
        
        self.rl_step_counter = 0
        return self._get_observations()

    def _get_observations(self):
        obs_list = []
        for i, g in enumerate(self.gliders):
            if g["done"]:
                obs_list.append(None)
                continue
            
            # For Tabular (index 1), we need discretized obs
            if i == 1:
                obs, laz, ldw = get_tabular_obs(self, g["phy_state"], g["aoa_idx"], g["bank_idx"], g["w_accel"], g["delta_w"], g["last_idx_az"], g["last_idx_dw"])
                g["last_idx_az"], g["last_idx_dw"] = laz, ldw
                obs_list.append(obs)
            else:
                # For Random (0) and DQN (2), we can just use the raw values
                obs_list.append(np.array([g["aoa_idx"], g["bank_idx"], g["w_accel"], g["delta_w"]], dtype=np.float32))
        return obs_list

    def step(self, actions):
        """
        Steps all gliders in parallel, advances wind once.
        """
        for i, g in enumerate(self.gliders):
            if g["done"]: continue
            
            action = actions[i]
            aoa_delta = (action // 3) - 1
            bank_delta = (action % 3) - 1
            control_changed = (aoa_delta != 0) or (bank_delta != 0)
            
            g["aoa_idx"] = np.clip(g["aoa_idx"] + aoa_delta, 0, config.AOA_BINS - 1)
            g["bank_idx"] = np.clip(g["bank_idx"] + bank_delta, 0, config.BANK_BINS - 1)
            
            if control_changed:
                v_tas, gamma, dchi_dt = self.physics_table_drag[g["aoa_idx"], g["bank_idx"]]
            else:
                v_tas, gamma, dchi_dt = self.physics_table[g["aoa_idx"], g["bank_idx"]]
            
            sum_w_accel = 0.0
            sum_delta_w = 0.0
            z_old = g["phy_state"][2]
            
            # Physics loop
            for _ in range(config.N_PHYS_PER_RL):
                x, y, z, chi = g["phy_state"]
                w_vec_start = self.wind_manager.get_wind(x, y, z) * config.WIND_AMPF
                
                dt = config.DT_RL / config.N_PHYS_PER_RL
                dx = (v_tas * np.cos(gamma) * np.cos(chi) + w_vec_start[0]) * dt
                dy = (v_tas * np.cos(gamma) * np.sin(chi) + w_vec_start[1]) * dt
                dz = (-v_tas * np.sin(gamma) + w_vec_start[2]) * dt
                
                g["phy_state"][0] = (x + dx) % config.DOMAIN_SIZE[0]
                g["phy_state"][1] = (y + dy) % config.DOMAIN_SIZE[1]
                g["phy_state"][2] += dz
                g["phy_state"][3] = (chi + dchi_dt * dt) % (2 * np.pi)
                
                w_vec_end = self.wind_manager.get_wind(*g["phy_state"][:3]) * config.WIND_AMPF
                sum_w_accel += (w_vec_end[2] - w_vec_start[2]) / dt
                
                side_vec = np.array([np.sin(chi), -np.cos(chi), 0])
                pos_r = g["phy_state"][:3] + (config.WINGSPAN / 2.0) * side_vec
                pos_r[:2] %= config.DOMAIN_SIZE[:2]
                pos_r[2] = np.clip(pos_r[2], 0, config.DOMAIN_SIZE[2] - 0.01)
                pos_l = g["phy_state"][:3] - (config.WINGSPAN / 2.0) * side_vec
                pos_l[:2] %= config.DOMAIN_SIZE[:2]
                pos_l[2] = np.clip(pos_l[2], 0, config.DOMAIN_SIZE[2] - 0.01)
                
                w_r = self.wind_manager.get_wind(*pos_r)[2] * config.WIND_AMPF
                w_l = self.wind_manager.get_wind(*pos_l)[2] * config.WIND_AMPF
                sum_delta_w += (w_r - w_l)
                
                if (g["phy_state"][2] <= config.DOMAIN_SIZE[2] * 0.1) or (g["phy_state"][2] >= config.DOMAIN_SIZE[2] * 0.9):
                    g["done"] = True
                    g["h_final"] = g["phy_state"][2]
                    break
            
            g["w_accel"] = sum_w_accel / config.N_PHYS_PER_RL
            g["delta_w"] = sum_delta_w / config.N_PHYS_PER_RL
            
            # Reward (Optional for evaluation but kept for consistency)
            current_uz = self.wind_manager.get_wind(*g["phy_state"][:3])[2] * config.WIND_AMPF
            height_change = g["phy_state"][2] - z_old
            g["ep_reward"] += current_uz + 5* g["w_accel"] + config.REWARD_SURVIVE + height_change * config.REWARD_LAMBDA
            g["h_final"] = g["phy_state"][2]

        # Sync Wind Step
        self.rl_step_counter += 1
        if self.rl_step_counter % config.RL_STEPS_PER_FRAME == 0:
            if not self.wind_manager.step_time():
                for g in self.gliders: g["done"] = True
        
        return self._get_observations(), [g["done"] for g in self.gliders]

    # Mock _apply_hysteresis for get_tabular_obs (it needs self as env)
    def _apply_hysteresis(self, value, bins, last_idx):
        if last_idx is None: return np.digitize(value, bins)
        new_idx = np.digitize(value, bins)
        if new_idx == last_idx: return last_idx
        target_bin_idx = last_idx if new_idx > last_idx else new_idx
        threshold = bins[target_bin_idx]
        margin = abs(threshold) * config.HYSTERESIS_PCT
        if new_idx > last_idx:
            return new_idx if value > threshold + margin else last_idx
        else:
            return new_idx if value < threshold - margin else last_idx

def run_eval(env, q_table, n_episodes=500, policy_type="trained"):
    """执行评估循环 (Tabular 策略)"""
    all_rewards = []
    all_climb_heights = []
    for ep in range(n_episodes):
        # 使用与训练相同的随机起始时间逻辑
        random_reset_time = np.random.randint(config.RESET_TIME_MIN, config.RESET_TIME_MAX)
        state, info = env.reset(options={"resettime": random_reset_time})
        h_start = info["height"]
        
        ep_reward = 0
        done = False
        last_height = h_start
        
        while not done:
            if policy_type == "random":
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[tuple(state)])
                
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            last_height = info["height"]
            done = terminated or truncated
            
        all_rewards.append(ep_reward)
        all_climb_heights.append(last_height - h_start)
        
    return all_rewards, all_climb_heights

def plot_climb_results(rnd_climbs, exp_climbs, save_path="trainresult/climb_eval_result.png", show=True):
    """绘制爬升高度对比图 (Seaborn 样式, 2 策略对比)"""
    n_episodes = len(rnd_climbs)
    df = pd.DataFrame({
        'Climb': rnd_climbs + exp_climbs,
        'Policy': ['Random Policy'] * n_episodes + ['Trained Policy'] * n_episodes
    })

    plt.figure(figsize=(7, 7))
    sns.set_style("white")
    my_colors = {"Random Policy": "#7f7f7f", "Trained Policy": "#d62728"}

    # 1. 散点图
    ax = sns.stripplot(data=df, x='Policy', y='Climb', palette=my_colors, 
                      hue='Policy', jitter=False, alpha=0.4, size=4, legend=False)

    # 2. 箱线图
    sns.boxplot(data=df, x='Policy', y='Climb', width=0.2, 
                showfliers=False, 
                boxprops={'facecolor':'none', 'edgecolor':'black', 'linewidth':1.5},
                medianprops={'color':'#ff7f0e', 'linewidth':2})

    # 3. 显示平均值和方差
    policies = ['Random Policy', 'Trained Policy']
    data_list = [rnd_climbs, exp_climbs]

    for i, p in enumerate(policies):
        mean_val = np.mean(data_list[i])
        std_val = np.std(data_list[i])
        y_pos = max(data_list[i]) + (max(df['Climb']) - min(df['Climb'])) * 0.05
        text_str = f"$\mu={mean_val:.1f}$\n$\sigma={std_val:.1f}$"
        plt.text(i, y_pos, text_str, ha='center', va='bottom', 
                 fontsize=16, fontweight='bold', color=my_colors[p])
        
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylabel("Episodic Climb Height (m)", fontsize=20)
    plt.xlabel("")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()

def plot_multi_climb_results(results, save_path="trainresult/compare_eval_result.png"):
    """绘制爬升高度对比图 (3 策略)"""
    data = []
    for policy, climbs in results.items():
        for c in climbs:
            data.append({"Policy": policy, "Climb": c})
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 7))
    sns.set_style("white")
    my_colors = {"Random": "#7f7f7f", "Tabular Q": "#1f77b4", "DQN": "#d62728"}

    ax = sns.stripplot(data=df, x='Policy', y='Climb', palette=my_colors, 
                      hue='Policy', jitter=0.05, alpha=0.4, size=4, legend=False)

    sns.boxplot(data=df, x='Policy', y='Climb', width=0.2, 
                showfliers=False, 
                boxprops={'facecolor':'none', 'edgecolor':'black', 'linewidth':1.5},
                medianprops={'color':'#ff7f0e', 'linewidth':2})

    for i, policy in enumerate(["Random", "Tabular Q", "DQN"]):
        climbs = results[policy]
        mean_val = np.mean(climbs)
        std_val = np.std(climbs)
        y_pos = df['Climb'].max() + (df['Climb'].max() - df['Climb'].min()) * 0.05
        text_str = f"$\mu={mean_val:.1f}$\n$\sigma={std_val:.1f}$"
        plt.text(i, y_pos, text_str, ha='center', va='bottom', 
                 fontsize=18, fontweight='bold', color=my_colors[policy])
        
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylabel("Episodic Climb Height (m)", fontsize=18)
    plt.xlabel("")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # --- 1. Load Models ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Tabular Q-Table
    if os.path.exists(config.SAVE_PATH):
        with open(config.SAVE_PATH, "rb") as f:
            q_table = pickle.load(f)
        print(f"Loaded Tabular Q-Table from {config.SAVE_PATH}")
    else:
        print("Warning: Q-Table not found. Tabular policy will be random.")
        q_table = np.zeros((config.AOA_BINS, config.BANK_BINS, 3, 3, 9))

    # --- 2. Initialize Evaluator ---
    h5_files = sorted(glob.glob(os.path.join(config.WIND_DIR, 'snapshots_s*.h5')), key=config.natural_key)
    evaluator = MultiGliderEvaluator(h5_files)
    
    # DQN Dimensions: [aoa_idx, bank_idx, w_accel, delta_w] -> 4, Actions -> 9
    state_dim = 4
    action_dim = 9

    dqn_model = QNetwork(state_dim, action_dim).to(device)
    if os.path.exists(config.DQN_SAVE_PATH):
        dqn_model.load_state_dict(torch.load(config.DQN_SAVE_PATH, map_location=device))
        dqn_model.eval()
        print(f"Loaded DQN Model from {config.DQN_SAVE_PATH}")
    else:
        print("Warning: DQN Model not found.")

    
    n_episodes = config.N_EVAL_EPISODES
    results = {"Random": [], "Tabular Q": [], "DQN": []}
    
    print(f"Starting parallel evaluation over {n_episodes} episodes...")
    import time
    start_time = time.time()

    for ep in range(n_episodes):
        reset_time = np.random.randint(config.RESET_TIME_MIN, config.RESET_TIME_MAX)
        obs_list = evaluator.reset_episode(reset_time)
        
        while True:
            actions = []
            # Random action
            actions.append(np.random.randint(0, 9))
            
            # Tabular action
            if obs_list[1] is not None:
                actions.append(np.argmax(q_table[tuple(obs_list[1])]))
            else:
                actions.append(0) # placeholder
                
            # DQN action
            if obs_list[2] is not None:
                s_norm = normalize_state(obs_list[2])
                state_t = torch.FloatTensor(s_norm).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = dqn_model(state_t).squeeze()
                    best_action = q_values.argmax().item()
                    # Hysteresis
                    if best_action != 4:
                        q_range = q_values.max() - q_values.min()
                        threshold = max(config.DQN_ACTION_MARGIN_MIN, config.DQN_ACTION_MARGIN_K * q_range)
                        if q_values[best_action] < q_values[4] + threshold:
                            best_action = 4
                    actions.append(best_action)
            else:
                actions.append(0) # placeholder
                
            obs_list, dones = evaluator.step(actions)
            if all(dones):
                break
        
        results["Random"].append(evaluator.gliders[0]["h_final"] - evaluator.gliders[0]["h_start"])
        results["Tabular Q"].append(evaluator.gliders[1]["h_final"] - evaluator.gliders[1]["h_start"])
        results["DQN"].append(evaluator.gliders[2]["h_final"] - evaluator.gliders[2]["h_start"])
        
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{n_episodes} complete. Time elapsed: {time.time() - start_time:.1f}s")

    print(f"\n--- Evaluation Results ---")
    for policy, climbs in results.items():
        print(f"{policy}: Mean Climb = {np.mean(climbs):.1f}m, Std = {np.std(climbs):.1f}m")
    
    plot_multi_climb_results(results)
    evaluator.wind_manager.close()
