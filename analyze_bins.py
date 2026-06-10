import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glider_discrete_simp import GliderEnv
import os
import glob
import config
import json

# --- 1. 配置环境 ---
h5_files = sorted(glob.glob(os.path.join(config.WIND_DIR, 'snapshots_s*.h5')))

if not h5_files:
    print(f"Error: No wind files found in '{config.WIND_DIR}' directory.")
    exit()

# 使用 memory_mode=False 进入“磁盘按需读取”模式 (低内存占用)
env = GliderEnv(h5_file_path=h5_files, polar_file_base=config.POLAR_BASE, memory_mode=False)

# --- 2. 运行随机策略收集数据 ---
N_EPISODES = 50  # 磁盘读取较慢，跑50个Episode足够统计
accel_data = []
delta_w_data = []
w_speed_data = []

print(f"开始运行随机策略收集传感器数据 (磁盘读取模式)...")
for ep in range(N_EPISODES):
    # 随机重置时间点以覆盖不同时刻的风场
    obs, info = env.reset(options={"resettime": np.random.randint(0, 300)})
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        accel_data.append(info["w_accel"])
        delta_w_data.append(info["delta_w"])

        # 计算 3D 风速向量的模 (即风速绝对值)
        w_vec = env.wind_manager.get_wind(*env.phy_state[:3]) * env.wind_ampf
        w_speed = np.linalg.norm(w_vec)
        w_speed_data.append(w_speed)

        done = terminated or truncated
    print(f"Episode {ep+1}/{N_EPISODES} finished.")

# --- 3. 统计与可视化 ---
accel_data = np.array(accel_data)
delta_w_data = np.array(delta_w_data)
w_speed_data = np.array(w_speed_data)

stats = {
    "w_accel": {
        "mean": float(np.mean(accel_data)),
        "std": float(np.std(accel_data))
    },
    "delta_w": {
        "mean": float(np.mean(delta_w_data)),
        "std": float(np.std(delta_w_data))
    },
    "wind_speed": {
        "mean": float(np.mean(w_speed_data)),
        "std": float(np.std(w_speed_data)),
        "rms": float(np.sqrt(np.mean(w_speed_data**2)))
    }
}

# 保存统计数据到 JSON
stats_path = os.path.join(config.BASE_DIR, "sensor_stats.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=4)
print(f"统计数据已保存至 {stats_path}")

def print_stats(name, data):
    print(f"\n--- {name} 统计结果 ---")
    print(f"均值: {np.mean(data):.4f}")
    print(f"标准差 (Std): {np.std(data):.4f}")
    print(f"分位数 [5%, 25%, 50%, 75%, 95%]:")
    print(np.percentile(data, [5, 25, 50, 75, 95]))

print_stats("Vertical Acceleration (w_accel)", accel_data)
print_stats("Wingtip Difference (delta_w)", delta_w_data)

# 打印 总风速 (3D Magnitude) 的 RMS
print(f"\n--- Total Wind Speed (Magnitude) 统计结果 (WIND_AMPF={config.WIND_AMPF}) ---")
print(f"风速绝对值的均方根 (RMS Speed): {stats['wind_speed']['rms']:.4f}")
print(f"均值: {stats['wind_speed']['mean']:.4f}")
print(f"标准差 (Std): {stats['wind_speed']['std']:.4f}")


# 绘图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(accel_data, kde=True, color="blue")
plt.title("w_accel Distribution")
# 标记 config 中的分箱线作为参考
for i, val in enumerate(config.BINS_W_ACCEL):
    plt.axvline(x=val, color='red', linestyle='--', label='Config Bin' if i==0 else "")
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(delta_w_data, kde=True, color="green")
plt.title("delta_w Distribution")
for val in config.BINS_DELTA_W:
    plt.axvline(x=val, color='red', linestyle='--')

plt.tight_layout()
save_path = os.path.join(config.TRAIN_RESULT_DIR, "sensor_distribution.png")
plt.savefig(save_path, dpi=300)
print(f"\n直方图已保存至 {save_path}")
plt.show()

env.close()
