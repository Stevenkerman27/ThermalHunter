import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import pickle
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from glider_discrete_simp import GliderEnv

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def simulate_with_env():
    # ================= 1. 配置参数 =================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wind_dir = os.path.join(base_dir, 'wind')
    h5_files = sorted(glob.glob(os.path.join(wind_dir, 'snapshots_s*.h5')), key=natural_key)
    
    Q_TABLE_DIR = "q_table"
    Q_TABLE_PATH = os.path.join(Q_TABLE_DIR, "q_table_ideal.pkl")
    #Q_TABLE_PATH = os.path.join(Q_TABLE_DIR, "q_table_v0.pkl")

    if not h5_files:
        print("错误：未找到风场文件。")
        return
    
    if not os.path.exists(Q_TABLE_PATH):
        print(f"错误：找不到 Q-table 文件 {Q_TABLE_PATH}")
        return

    CONFIG = {
        "polar_base": 'glider',
        "domain_size": (1000.0, 1000.0, 1000.0),
    }

    # ================= 2. 初始化环境与模型 =================
    env = GliderEnv(
        h5_file_path=h5_files,
        polar_file_base=CONFIG["polar_base"],
        domain_size=CONFIG["domain_size"],
    )

    with open(Q_TABLE_PATH, "rb") as f:
        q_table = pickle.load(f)
    print(f"成功加载 Q-table: {Q_TABLE_PATH}")

    obs, info = env.reset(options={"resettime": 80})
    start_h = info["height"]
    
    # ================= 3. 执行模拟 =================
    history = []      # 记录位置 [x, y, z]
    tas_history = []  # 记录真实空速
    uz_history = []
    reward_hst = []
    w_accels, delta_ws = [], []
    obs_bank_idxs, obs_w_accel_idxs, obs_delta_w_idxs = [], [], [] 
    aoa, bank = [], []
    
    max_steps = 1000
    print(f"开始模拟... dt_rl={env.dt_rl:.4f}s")

    for _ in range(max_steps):
        # 记录当前物理状态（位置）
        history.append(env.phy_state[:3].copy())

        # 根据当前观测选择动作
        action = np.argmax(q_table[tuple(obs)])
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 从 info 中提取物理数据 
        tas_history.append(info["tas"]) 
        uz_history.append(info["uz"])
        w_accels.append(info["w_accel"])
        delta_ws.append(info["delta_w"])
        aoa.append(info["control"][0])
        bank.append(info["control"][1])
        
        # 记录离散观察值和奖励
        obs_bank_idxs.append(obs[0])
        obs_w_accel_idxs.append(obs[1])
        obs_delta_w_idxs.append(obs[2])
        reward_hst.append(reward)

        if terminated or truncated:
            break

    print(f"模拟结束。总奖励: {np.sum(reward_hst):.2f}, 开始高度：{start_h}最终高度: {info['height']:.2f}m")

    # ================= 4. 绘图 =================
    _plot_all_results(
        np.array(history), 
        np.array(tas_history), 
        np.array(uz_history),
        np.array(reward_hst),
        np.array(w_accels), np.array(delta_ws), 
        np.array(obs_bank_idxs), np.array(obs_w_accel_idxs), np.array(obs_delta_w_idxs),
        np.array(aoa), np.array(bank),
        env, CONFIG["domain_size"]
    )
    
    env.close()

def _plot_all_results(history, tas, uz, reward_hst, w_accels, delta_ws, obs_bank_idxs, obs_w_accel_idxs, obs_delta_w_idxs, aoa, bank, env, domain_size):
    times = np.arange(len(tas)) * env.dt_rl

    # --- 图 1: 3D 轨迹图 ---
    fig1 = plt.figure(figsize=(10, 7))
    ax3d = fig1.add_subplot(111, projection='3d')
    
    # 核心修正：N个点生成N-1条线段
    # segments 的形状应该是 (N-1, 2, 3)
    points = history[:-1].reshape(-1, 1, 3)
    next_points = history[1:].reshape(-1, 1, 3)
    segments = np.concatenate([points, next_points], axis=1)
    
    # 过滤穿越边界的无效线段 (周期性边界条件)
    diffs = np.linalg.norm(segments[:, 0, :] - segments[:, 1, :], axis=1)
    valid_mask = diffs < (min(domain_size) * 0.5) 
    
    uz_for_segments = uz[:len(segments)] # 使用垂直风速
    
    lc = Line3DCollection(segments[valid_mask], cmap='viridis', 
                          norm=Normalize(vmin=uz.min(), vmax=uz.max()))
    lc.set_array(uz_for_segments[valid_mask])
    
    ax3d.add_collection3d(lc)
    fig1.colorbar(lc, ax=ax3d, label='Vertical Wind Speed(m/s)', pad=0.1)
    
    ax3d.set_xlim(0, domain_size[0]); ax3d.set_ylim(0, domain_size[1]); ax3d.set_zlim(0, domain_size[2])
    ax3d.set_title('Glider Trajectory (Colored by TAS)')

    # --- 图 2: 特征分析图 ---
    fig2, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
    (ax1, ax2, ax3, ax4, ax5) = axes

    # 1. 风加速度
    ax1.plot(times, w_accels, color='steelblue', label='Actual $w_{accel}$')
    ax1_tw = ax1.twinx()
    ax1_tw.step(times, obs_w_accel_idxs, where='post', color='green', alpha=0.5, label='Obs Index')
    ax1.set_ylabel('Accel ($m/s^2$)'); ax1.set_title("Wind Vertical Acceleration")
    ax1.grid(True, alpha=0.3)

    # 2. 翼尖风速差
    ax2.plot(times, delta_ws, color='forestgreen', label='Actual $\delta_w$')
    ax2_tw = ax2.twinx()
    ax2_tw.step(times, obs_delta_w_idxs, where='post', color='red', alpha=0.5, label='Obs Index')
    ax2.set_ylabel('Diff ($m/s$)'); ax2.set_title("Wingtip Wind Difference")
    ax2.grid(True, alpha=0.3)

    # 3. 奖励
    ax5.plot(times, reward_hst, color='purple')
    ax5.set_ylabel('Step Reward'); ax5.set_title("Reward per Step")
    ax5.grid(True, alpha=0.3)

    # 4. 控制量 (角度)
    ax3.step(times, np.rad2deg(bank), where='post', color='red', label='Bank')
    ax3_tw = ax3.twinx()
    ax3_tw.step(times, np.rad2deg(aoa), where='post', color='green', label='AoA')
    
    ax3.set_ylabel('Bank (red)'); ax3_tw.set_ylabel('AoA (green)')
    ax3.set_title("Control Inputs")
    ax3.grid(True, alpha=0.3)

    # 5. 真实空速 (TAS)
    ax4.step(times, tas, where='post', color='black', linewidth=1.5)
    ax4.set_ylabel('TAS (m/s)'); ax4.set_xlabel('Time (s)')
    ax4.set_title("True Airspeed (Corrected)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_with_env()