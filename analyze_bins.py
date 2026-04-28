import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glider_discrete_simp import GliderEnv, RBWindField
import os
import glob
import h5py
import config

# --- 1. Monkeypatch: 让风场管理器进入“磁盘按需读取”模式 (低内存占用) ---
def lazy_open_resources(self):
    self.all_data = [] 
    self.files = [] # 确保 files 列表存在
    self.dsets_list = []
    self.t_offsets = [0]
    self.all_sim_times = []
    
    for path in self.h5_paths:
        f = h5py.File(path, 'r')
        self.files.append(f) # 保持文件开启状态
        dset_group = {
            'ux': f['tasks/ux'], # 注意：这里去掉了 [:]，不加载入内存
            'uy': f['tasks/uy'],
            'uz': f['tasks/uz'],
            'buoyancy': f['tasks/buoyancy']
        }
        self.dsets_list.append(dset_group)
        file_times = f['tasks/ux'].dims[0]['sim_time'][:]
        self.all_sim_times.extend(file_times)
        self.t_offsets.append(self.t_offsets[-1] + len(file_times))
    
    self.all_sim_times = np.array(self.all_sim_times)
    self.max_t_idx = len(self.all_sim_times) - 1
    if len(self.all_sim_times) > 1:
        self.dt_phy = self.all_sim_times[1] - self.all_sim_times[0]
    
    first_shape = self.dsets_list[0]['ux'].shape
    # RBWindField 内部索引顺序: (t, x, y, z)
    self.space_range = [first_shape[1], first_shape[2], first_shape[3]]
    print(f"Lazy WindField (Disk-based) initialized. Total steps: {self.max_t_idx + 1}")

# 替换原始方法
RBWindField._open_resources = lazy_open_resources

# --- 2. 配置环境 ---
h5_files = sorted(glob.glob(os.path.join(config.WIND_DIR, 'snapshots_s*.h5')))

if not h5_files:
    print(f"Error: No wind files found in '{config.WIND_DIR}' directory.")
    exit()

env = GliderEnv(h5_file_path=h5_files, polar_file_base=config.POLAR_BASE)

# --- 3. 运行随机策略收集数据 ---
N_EPISODES = 20  # 磁盘读取较慢，跑20个Episode足够统计
accel_data = []
delta_w_data = []

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
        done = terminated or truncated
    print(f"Episode {ep+1}/{N_EPISODES} finished.")

# --- 4. 统计与可视化 ---
accel_data = np.array(accel_data)
delta_w_data = np.array(delta_w_data)

def print_stats(name, data):
    print(f"\n--- {name} 统计结果 ---")
    print(f"均值: {np.mean(data):.4f}")
    print(f"标准差 (Std): {np.std(data):.4f}")
    print(f"分位数 [5%, 25%, 50%, 75%, 95%]:")
    print(np.percentile(data, [5, 25, 50, 75, 95]))

print_stats("Vertical Acceleration (w_accel)", accel_data)
print_stats("Wingtip Difference (delta_w)", delta_w_data)

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
