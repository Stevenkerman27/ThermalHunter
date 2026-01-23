import numpy as np
import os
import glob
from glider_discrete import RBWindField
import re

def diagnose_wind_file():
    wind_dir = os.path.join(os.path.dirname(__file__), 'wind')
    raw_files = glob.glob(os.path.join(wind_dir, 'snapshots_s*.h5'))
    
    # 修复排序逻辑
    h5_files = sorted(raw_files, key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)])
    
    if not h5_files:
        print("未找到任何风场文件！")
        return
    
    # 2. 初始化 (RBWindField 内部会读取 sim_time 标尺)
    domain_size = (1000.0, 1000.00, 1000.0)
    try:
        wf = RBWindField(h5_files, domain_size=domain_size)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    total_steps = wf.max_t_idx + 1
    print(f"总时间步数: {total_steps}")
    # 调整表头，增加 Sim Time 列
    print("-" * 80)
    print(f"{'Global T':<10} | {'File':<6} | {'Sim Time':<10} | {'Max U':<10} | {'Max V':<10} | {'Max W':<10} | {'Status'}")
    print("-" * 80)

    non_zero_frames = 0
    
    for t_global in range(total_steps):
        try:
            # 获取物理时间
            sim_time = wf.all_sim_times[t_global] 
            
            # 定位子文件和局部索引 (为了读取原始数据)
            file_idx = 0
            for i in range(len(wf.t_offsets) - 1):
                if wf.t_offsets[i] <= t_global < wf.t_offsets[i+1]:
                    file_idx = i
                    break
            local_t = t_global - wf.t_offsets[file_idx]
            
            # 读取数据
            current_dsets = wf.dsets_list[file_idx]
            u_raw = current_dsets['ux'][local_t]
            v_raw = current_dsets['uy'][local_t]
            w_raw = current_dsets['uz'][local_t]
            
            max_u, max_v, max_w = np.max(np.abs(u_raw)), np.max(np.abs(v_raw)), np.max(np.abs(w_raw))
            
            status = "ACTIVE" if (max_u > 1e-5 or max_v > 1e-5 or max_w > 1e-5) else "EMPTY"
            if status == "ACTIVE": non_zero_frames += 1
            
            # 打印逻辑
            if status == "ACTIVE" or t_global % 10 == 0 or t_global == total_steps - 1:
                file_name = f"s{file_idx+1}"
                # 格式化输出，sim_time 保留 2 位小数
                print(f"{t_global:<10} | {file_name:<6} | {sim_time:<10.2f} | {max_u:<10.4f} | {max_v:<10.4f} | {max_w:<10.4f} | {status}")
                
        except Exception as e:
            print(f"{t_global:<10} | 读取出错: {e}")

    print("-" * 80)
    print(f"诊断完成。总物理时长: {wf.all_sim_times[-1] - wf.all_sim_times[0]:.2f} (units)")
    wf.close()

if __name__ == "__main__":
    diagnose_wind_file()