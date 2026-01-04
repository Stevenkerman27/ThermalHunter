import numpy as np
import os
import glob
from glider_discrete import RBWindField

#诊断是否读取出有效的风场

def diagnose_wind_file():
    # 1. 自动查找文件
    wind_dir = os.path.join(os.path.dirname(__file__), 'wind')
    h5_files = glob.glob(os.path.join(wind_dir, '*.h5'))
    
    if not h5_files:
        print("未找到风场文件！")
        return
    
    h5_path = h5_files[0]
    print(f"正在诊断文件: {h5_path}")
    
    # 2. 初始化
    domain_size = (100.0, 100.0, 100.0) # 这里的尺寸只影响坐标映射，不影响原始数据统计
    try:
        wf = RBWindField(h5_path, domain_size=domain_size)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    total_steps = wf.max_t_idx + 1
    print(f"总时间步数: {total_steps}")
    print("-" * 50)
    print(f"{'Time Idx':<10} | {'Max U':<10} | {'Max V':<10} | {'Max W':<10} | {'Status'}")
    print("-" * 50)

    # 3. 循环扫描每一帧
    non_zero_frames = 0
    
    # 为了速度，直接读取 HDF5 数据集进行统计，而不是通过插值函数
    # 这样可以确认是"没读到"还是"数据本身就是0"
    for t in range(total_steps):
        try:
            # 直接切片读取原始 Grid 数据
            # 注意：这里假设数据形状是 (t, x, y, z)
            u_raw = wf.dsets['ux'][t]
            v_raw = wf.dsets['uy'][t]
            w_raw = wf.dsets['uz'][t]
            
            # 计算绝对值的最大值
            max_u = np.max(np.abs(u_raw))
            max_v = np.max(np.abs(v_raw))
            max_w = np.max(np.abs(w_raw))
            
            status = "EMPTY"
            if max_u > 1e-5 or max_v > 1e-5 or max_w > 1e-5:
                status = "ACTIVE"
                non_zero_frames += 1
            
            # 只打印 非零帧 或者 某些特定帧（避免刷屏）
            if status == "ACTIVE" or t % 10 == 0 or t == total_steps - 1:
                print(f"{t:<10} | {max_u:<10.4f} | {max_v:<10.4f} | {max_w:<10.4f} | {status}")
                
        except Exception as e:
            print(f"{t:<10} | 读取出错: {e}")

    print("-" * 50)
    if non_zero_frames == 0:
        print("警告：整个文件中所有时间步的风速数据似乎都是 0！")
        print("建议：请检查 HDF5 文件源数据是否正确生成。")
    else:
        print(f"诊断完成。共发现 {non_zero_frames} 个有效风场帧。")
        print("你可以使用 wf.reset(t_index=N) 来指定加载这些有数据的帧。")

    wf.close()

if __name__ == "__main__":
    diagnose_wind_file()