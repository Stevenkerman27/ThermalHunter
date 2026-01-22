import os
import numpy as np
import h5py
import glob
from glider_discrete import RBWindField  # 假设你的原文件名为 glider_discrete.py

def test_wind_reading():
    # 1. 路径设置
    wind_dir = os.path.join(os.path.dirname(__file__), 'wind')
    
    # 2. 搜集所有 snapshots 文件并排序
    h5_files = sorted(glob.glob(os.path.join(wind_dir, 'snapshots_s*.h5')))
    if not h5_files:
        print("[Error] 未找到任何 HDF5 文件")
        return
    print(f"[Test] 找到文件列表: {[os.path.basename(f) for f in h5_files]}")

    # 3. 初始化（传入整个列表）
    domain_size = (64, 64, 64) 
    wind_field = RBWindField(h5_files, domain_size=domain_size)

    # 4. 验证跨文件索引逻辑
    # 假设每个文件有 100 个 step，测试读取第 150 个 step（应指向 s2）
    test_global_t = 150 
    actual_t = wind_field.reset(test_global_t)
    print(f"\n[Check 1] 跨文件跳转:")
    print(f"  - 设定全局 T: {test_global_t}")
    print(f"  - 实际定位文件索引: {wind_field.current_file_idx} (s{wind_field.current_file_idx+1})")
    print(f"  - 局部文件内索引: {wind_field.local_t_idx}")

    # 4. 验证内部状态 (Deep Check)
    print(f"\n[Check 2] 内部状态验证:")
    print(f"  - 时间步总数 (Max T): {wind_field.max_t_idx}")

    # 5. 测试 Reset
    t_idx = wind_field.reset(40)
    print(f"  - Reset 后当前时间索引: {t_idx}")

    # 6. 测试 get_wind (核心逻辑)
    print(f"\n[Check 3] 风速读取测试 (get_wind):")
    
    test_points = [
        # (名称, x, y, z)
        ("原点 (0,0,0)", 0.0, 0.0, 0.0),
        ("域中心", 0.5, 0.5, 0.5),
        ("边界内一点", 0.97, 0.97, 0.97),
        ("边界内一点", 1, 1, 1)
    ]

    for name, x, y, z in test_points:
        wind_vec = wind_field.get_wind(x, y, z)
        print(f"  测试点 [{name}]:")
        print(f"    输入坐标: ({x:.3f}, {y:.3f}, {z:.3f})")
        print(f"    输出风速: ux={wind_vec[0]:.4f}, uy={wind_vec[1]:.4f}, uz={wind_vec[2]:.4f}")
        
        # 简单的数据合理性检查
        if np.isnan(wind_vec).any():
            print("    [Error] 检测到 NaN 值！")
        else:
            print("    [Pass] 数据格式正常")

    # 7. 清理
    wind_field.close()
    print("-" * 40)
    print("测试完成。")

if __name__ == "__main__":
    test_wind_reading()