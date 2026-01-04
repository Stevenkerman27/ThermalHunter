import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# 引用你的模块
from glider_discrete import RBWindField, GliderPhysics

def simulate_constant_control():
    # ================= 配置参数 =================
    # 1. 文件路径
    H5_PATH = os.path.join(os.path.dirname(__file__), 'wind/snapshots_s1.h5') # 请修改为你的实际路径
    POLAR_BASE = 'glider' # 假设你的气动数据前缀是 glider (即存在 glider_DegenGeom.polar)
    
    # 2. 物理环境定义
    # 假设我们模拟 1km 的立方体
    DOMAIN_SIZE_M = np.array([1000.0, 1000.0, 1000.0]) 
    
    # 3. 风速缩放 (关键步骤)
    # 假设 Dedalus 输出的无量纲 w 最大值约为 80，我们希望物理最大 w 为 5 m/s
    # Scale = 5.0 / 80.0 = 0.0625
    WIND_SCALE = 1000 

    # 4. 初始状态
    # [x, y, z, V_tas, gamma, chi]
    # 从 (500, 500, 900) 开始，速度 15m/s，水平飞行，朝向北方 (chi=pi/2)
    init_state = np.array([500.0, 500.0, 900.0, 15.0, 0.0, np.pi/2])

    # 5. 控制输入 (固定)
    aoa_deg = 3
    bank_deg = 5
    
    control = np.array([np.deg2rad(aoa_deg), np.deg2rad(bank_deg)])

    # ================= 初始化 =================
    print(f"正在初始化风场: {H5_PATH}...")
    try:
        # 注意：这里 domain_size 传给 WindField 主要是为了内部记录，
        # 但 get_wind 依然需要 0-1 的输入，我们在循环里手动转换更稳妥。
        wind_field = RBWindField(H5_PATH, domain_size=DOMAIN_SIZE_M)

        t_idx = wind_field.reset(40)
        print(f"当前风场时间索引: {t_idx}")
    except Exception as e:
        print(f"[Error] 风场加载失败: {e}")
        return

    try:
        physics = GliderPhysics(POLAR_BASE, mass=2)
        print("气动数据加载成功。")
    except Exception as e:
        print(f"[Error] 气动数据加载失败 (请确保有 .polar 文件): {e}")
        wind_field.close()
        return

    # ================= 主循环 =================
    dt = 0.1  # 模拟步长 (秒)
    max_time = 300 # 最多模拟 300 秒
    steps = int(max_time / dt)
    
    history = []
    velocities = []
    current_state = init_state.copy()
    
    print(f"开始模拟... (AoA={aoa_deg}°, Bank={bank_deg}°)")

    for i in range(steps):
        # 1. 记录轨迹
        pos = current_state[:3]
        v_tas = current_state[3]
        history.append(pos)
        velocities.append(v_tas)

        # 2. 坐标归一化 (Meters -> 0..1)
        # 必须加上防止除0的保护，虽然这里 domain固定是1000
        norm_pos = pos / DOMAIN_SIZE_M
        
        # 3. 边界检查 (简单版)
        if np.any(norm_pos < 0) or np.any(norm_pos > 1):
            print(f"Step {i}: 滑翔机飞出边界，停止模拟。Pos: {pos}")
            break
        
        # 4. 读取风速 (无量纲)
        # 注意：这里调用的是修改过(含clip)的 get_wind
        raw_wind = wind_field.get_wind(*norm_pos) 
        
        # 5. 风速物理化 (Scaling)
        real_wind = raw_wind * WIND_SCALE
        
        # 6. 物理积分
        # current_state 更新
        current_state = physics.integration_step(current_state, control, real_wind, dt)

        # 7. 接地检查
        if current_state[2] <= 0:
            print(f"Step {i}: 滑翔机落地。")
            break
    history = np.array(history)       # 将列表转为 numpy 数组
    velocities = np.array(velocities) # 将列表转为 numpy 数组

    # ================= 可视化 (带速度颜色映射) =================
    if len(history) == 0:
        print("没有轨迹数据。")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- 核心修改：创建彩色线条 ---
    
    # 1. 准备数据点对：(x, y, z) -> (next_x, next_y, next_z)
    points = history.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 2. 设置颜色映射 (Colormap)
    # 使用 'jet' (蓝-青-黄-红) 或 'viridis'
    # 这里的 norm 决定了颜色的范围。你可以自动设为 min/max，也可以手动指定范围方便观察
    norm = Normalize(vmin=np.min(velocities), vmax=np.max(velocities))
    cmap = plt.get_cmap('jet')

    # 3. 创建 Line3DCollection 对象
    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(velocities[:-1]) # 设置颜色依据的数据
    lc.set_linewidth(2)           # 线宽

    # 4. 添加到绘图区
    ax.add_collection3d(lc)
    
    # 5. 添加颜色条 (Colorbar) 以便读数
    cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('True Airspeed (m/s)')

    # --- 辅助元素 ---
    
    # 绘制起点和终点
    ax.scatter(history[0, 0], history[0, 1], history[0, 2], c='black', marker='o', s=50, label='Start')
    ax.scatter(history[-1, 0], history[-1, 1], history[-1, 2], c='black', marker='x', s=50, label='End')

    # 关键：Line3DCollection 不会自动更新坐标轴范围，必须手动设置！
    ax.set_xlim(history[:,0].min(), history[:,0].max())
    ax.set_ylim(history[:,1].min(), history[:,1].max())
    ax.set_zlim(0, 1000)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Glider Path (Color by Airspeed)\nAoA={aoa_deg}, Bank={bank_deg}')
    
    plt.show()

if __name__ == "__main__":
    simulate_constant_control()