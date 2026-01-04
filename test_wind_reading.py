import os
import numpy as np
import h5py
import glob
from glider_discrete import RBWindField  # 假设你的原文件名为 glider_discrete.py

def test_wind_reading():
    # 1. 路径设置
    wind_dir = os.path.join(os.path.dirname(__file__), 'wind')
    if not os.path.exists(wind_dir):
        os.makedirs(wind_dir)

    # 2. 查找或生成数据
    h5_files = glob.glob(os.path.join(wind_dir, '*.h5'))
    h5_path = h5_files[0]
    print(f"[Test] 找到现有文件: {h5_path}")

    # 3. 初始化参数
    # 假设物理域大小为 100m x 100m x 100m，方便验证 Scale
    domain_size = (64, 64, 64) 
    
    print("-" * 40)
    print("正在初始化 RBWindField...")
    try:
        wind_field = RBWindField(h5_path, domain_size=domain_size)
    except Exception as e:
        print(f"[Error] 初始化失败: {e}")
        return

    # 4. 验证内部状态 (Deep Check)
    print(f"\n[Check 1] 内部状态验证:")
    print(f"  - 时间步总数 (Max T): {wind_field.max_t_idx}")
    print(f"  - 缩放比例 (Scales): {wind_field.scales}")
    
    # 验证 Scale 计算逻辑: Scale = Grid_Size / Domain_Size
    # 如果是伪数据 (32, 32, 32) 和 域 (100, 100, 100)，Scale 应该是 0.32
    expected_scale_x = wind_field.dsets['ux'].shape[1] / domain_size[0]
    print(f"  - 预期 Scale X: {expected_scale_x:.4f} (实际: {wind_field.scales[0]:.4f})")

    # 5. 测试 Reset
    t_idx = wind_field.reset(40)
    print(f"  - Reset 后当前时间索引: {t_idx}")

    # 6. 测试 get_wind (核心逻辑)
    print(f"\n[Check 2] 风速读取测试 (get_wind):")
    
    test_points = [
        # (名称, x, y, z)
        ("原点 (0,0,0)", 0.0, 0.0, 0.0),
        ("域中心", 0.5, 0.5, 0.5),
        ("边界内一点", 0.97, 0.97, 0.97),
        ("边界内一点", 1, 1, 1)
    ]

    for name, x, y, z in test_points:
        wind_vec = wind_field.get_wind(x, y, z, 1000)
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