import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from glider_discrete import RBWindField, GliderPhysics

def simulate_constant_control():
    # ================= 配置参数 =================
    wind_dir = os.path.join(os.path.dirname(__file__), 'wind')
    h5_files = sorted(glob.glob(os.path.join(wind_dir, 'snapshots_s*.h5')))
    
    if not h5_files:
        print("错误：未找到风场文件。")
        return

    POLAR_BASE = 'glider' 
    DOMAIN_SIZE_M = np.array([800.0, 800.0, 800.0]) 
    WIND_AMPF = 20.0  
    B = 2.0           

    init_state = np.array([600.0, 200.0, 600.0, 12.0, 0.0, np.pi/2], dtype=np.float32)

    aoa_deg = 2
    bank_deg = 2
    control = np.array([np.deg2rad(aoa_deg), np.deg2rad(bank_deg)], dtype=np.float32)

    wind_field = RBWindField(h5_files, domain_size=DOMAIN_SIZE_M)
    wind_field.reset(0)
    dt = wind_field.dt_phy

    try:
        physics = GliderPhysics(POLAR_BASE, mass=2)
    except Exception as e:
        print(f"[Error] 气动数据加载失败: {e}")
        wind_field.close()
        return

    # ================= 主循环 =================
    max_time = 300 
    steps = int(max_time / dt)
    
    history = []
    velocities = []
    w_accels = []
    delta_ws = []
    
    current_state = init_state.copy()
    prev_w_z = wind_field.get_wind(*current_state[:3])[2] * WIND_AMPF

    print(f"开始模拟... 使用 dt={dt:.4f}s")

    for i in range(steps):
        pos = current_state[:3]
        v_tas = current_state[3]
        chi = current_state[5]
        
        history.append(pos.copy())
        velocities.append(v_tas)

        # 1. 计算 delta_w
        side_vec = np.array([np.sin(chi), -np.cos(chi), 0])
        pos_right = pos + (B / 2.0) * side_vec
        pos_left  = pos - (B / 2.0) * side_vec
        
        w_right_z = wind_field.get_wind(*pos_right)[2] * WIND_AMPF
        w_left_z  = wind_field.get_wind(*pos_left)[2] * WIND_AMPF
        delta_ws.append(w_right_z - w_left_z)

        # 2. 计算 w_accel
        curr_w_z = wind_field.get_wind(*pos)[2] * WIND_AMPF
        w_accel = (curr_w_z - prev_w_z) / dt
        w_accels.append(w_accel)
        prev_w_z = curr_w_z

        if (pos[2] <= 0 or np.any(pos < 0) or np.any(pos > DOMAIN_SIZE_M)):
            break
        
        current_state = physics.integration_step(
                    current_state, control, wind_field, WIND_AMPF, dt
                )

        if not wind_field.step_time():
            break

    # ================= 数据统计与转换 =================
    history = np.array(history)
    velocities = np.array(velocities)
    w_accels = np.array(w_accels)
    delta_ws = np.array(delta_ws)
    times = np.arange(len(history)) * dt

    if len(history) < 2:
        wind_field.close()
        return

    print("\n--- 轨迹风场特性统计 ---")
    print(f"w_accel 均值: {w_accels.mean():.4f} m/s², 范围: [{w_accels.min():.4f}, {w_accels.max():.4f}]")
    print(f"delta_w 均值: {delta_ws.mean():.4f} m/s, 范围: [{delta_ws.min():.4f}, {delta_ws.max():.4f}]")

    # ================= 可视化 =================
    
    # 图 1: 3D 轨迹
    fig1 = plt.figure(figsize=(10, 7))
    ax3d = fig1.add_subplot(111, projection='3d')
    points = history.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=velocities.min(), vmax=velocities.max())
    lc = Line3DCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(velocities[:-1])
    lc.set_linewidth(2)
    ax3d.add_collection3d(lc)
    fig1.colorbar(lc, ax=ax3d, label='True Airspeed (m/s)', pad=0.1)
    ax3d.set_xlim(0, DOMAIN_SIZE_M[0]); ax3d.set_ylim(0, DOMAIN_SIZE_M[1]); ax3d.set_zlim(0, DOMAIN_SIZE_M[2])
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Height (m)')
    ax3d.set_title('Glider Trajectory')

    # 图 2: 风场统计数据曲线
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 绘制 w_accel
    ax1.plot(times, w_accels, color='steelblue', lw=1.5, label='$w_{accel}$')
    ax1.axhline(w_accels.mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
    ax1.set_ylabel('Vertical Accel ($m/s^2$)')
    ax1.set_title('Dynamic Wind Characteristics Analysis')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # 绘制 delta_w
    ax2.plot(times, delta_ws, color='indianred', lw=1.5, label='$\delta_w$')
    ax2.axhline(delta_ws.mean(), color='blue', linestyle='--', alpha=0.7, label='Mean')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Wingtip Wind Diff ($m/s$)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    
    wind_field.close()

if __name__ == "__main__":
    simulate_constant_control()