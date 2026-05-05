import numpy as np
import os
import glob
import re
from glider_discrete_simp import GliderEnv
import config

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def test_sink_rate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wind_dir = os.path.join(base_dir, 'wind')
    h5_files = sorted(glob.glob(os.path.join(wind_dir, 'snapshots_s*.h5')), key=natural_key)

    env = GliderEnv(h5_file_path=h5_files, polar_file_base="glider", random_init=False, memory_mode=False)

    # 强制获取 (1, 1) 索引下的物理参数
    aoa_idx, bank_idx = 1, 1
    
    # 正常状态
    v_steady, gamma_steady, _ = env.physics_table[aoa_idx, bank_idx]
    sink_steady = v_steady * np.sin(gamma_steady)
    
    # 高阻力状态
    v_drag, gamma_drag, _ = env.physics_table_drag[aoa_idx, bank_idx]
    sink_drag = v_drag * np.sin(gamma_drag)
    
    print(f"\n--- Physical Parameters Comparison (State AoA_idx=1, Bank_idx=1) ---")
    print(f"Normal: TAS = {v_steady:.4f} m/s, Gamma = {np.degrees(gamma_steady):.4f} deg, Sink Rate = {sink_steady:.4f} m/s")
    print(f"Drag+ : TAS = {v_drag:.4f} m/s, Gamma = {np.degrees(gamma_drag):.4f} deg, Sink Rate = {sink_drag:.4f} m/s")
    
    diff_sink = sink_drag - sink_steady
    print(f"Sink Rate Increase: {diff_sink:.4f} m/s ({diff_sink/sink_steady*100:.1f}%)")

    env.close()

if __name__ == "__main__":
    test_sink_rate()
