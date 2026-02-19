import numpy as np
import gymnasium as gym
from gymnasium import spaces
import h5py
import re
import os
import pandas as pd
from scipy.interpolate import interp1d
from numba import njit

@njit
def trilinear(cube, dx, dy, dz):
    c0 = cube[0] * (1 - dx) + cube[1] * dx
    c1 = c0[0] * (1 - dy) + c0[1] * dy
    return c1[0] * (1 - dz) + c1[1] * dz

class RBWindField:
    def __init__(self, h5_paths, domain_size=(1000, 1000, 1000)):
        def natural_key(string_):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

        if isinstance(h5_paths, list):
            self.h5_paths = sorted(h5_paths, key=natural_key) 
        else:
            self.h5_paths = [h5_paths]

        self.domain_size = np.array(domain_size, dtype=np.float32)
        
        self.files = []
        self.dsets_list = [] 
        self.t_offsets = [0] 
        
        self.t_axis, self.x_axis, self.y_axis, self.z_axis = 0, 1, 2, 3
        self.space_range = [0, 0, 0]

        self.global_t_idx = 0
        self.current_file_idx = 0
        self.local_t_idx = 0
        self.all_sim_times = [] 
        self.dt_phy = 0.0

        self._open_resources()

    def _open_resources(self):
        for path in self.h5_paths:
            f = h5py.File(path, 'r')
            self.files.append(f)
            dset_group = {k: f['tasks'][k] for k in ['ux', 'uy', 'uz','buoyancy']}
            self.dsets_list.append(dset_group)
            
            file_times = dset_group['ux'].dims[0]['sim_time'][:]
            self.all_sim_times.extend(file_times)
            
            file_t_steps = len(file_times)
            self.t_offsets.append(self.t_offsets[-1] + file_t_steps)
        
        self.all_sim_times = np.array(self.all_sim_times)
        self.max_t_idx = len(self.all_sim_times) - 1
        
        if len(self.all_sim_times) > 1:
            self.dt_phy = self.all_sim_times[1] - self.all_sim_times[0]
        
        first_shape = self.dsets_list[0]['ux'].shape
        self.space_range[0] = first_shape[self.x_axis]
        self.space_range[1] = first_shape[self.y_axis]
        self.space_range[2] = first_shape[self.z_axis]
        
        print(f"WindField initialized. dt_phy: {self.dt_phy:.4f}, Total steps: {self.max_t_idx + 1}")

    def reset(self, t_index=0):
        self.global_t_idx = min(t_index, self.max_t_idx)
        self._update_file_pointers()
        return self.global_t_idx

    def _update_file_pointers(self):
        for i in range(len(self.t_offsets) - 1):
            if self.t_offsets[i] <= self.global_t_idx < self.t_offsets[i+1]:
                self.current_file_idx = i
                self.local_t_idx = self.global_t_idx - self.t_offsets[i]
                break

    def step_time(self):
        if self.global_t_idx < self.max_t_idx:
            self.global_t_idx += 1
            self._update_file_pointers()
            return True
        return False

    def get_wind(self, x, y, z):
        fx = np.clip((x / self.domain_size[0]) * (self.space_range[0] - 1), 0, self.space_range[0] - 1.00001)
        fy = np.clip((y / self.domain_size[1]) * (self.space_range[1] - 1), 0, self.space_range[1] - 1.00001)
        fz = np.clip((z / self.domain_size[2]) * (self.space_range[2] - 1), 0, self.space_range[2] - 1.00001)

        ix0, iy0, iz0 = int(fx), int(fy), int(fz)
        dx, dy, dz = fx - ix0, fy - iy0, fz - iz0

        slices = (slice(self.local_t_idx, self.local_t_idx + 1), slice(ix0, ix0 + 2), slice(iy0, iy0 + 2), slice(iz0, iz0 + 2))
        dsets = self.dsets_list[self.current_file_idx]

        return np.array([
            trilinear(dsets['ux'][slices].squeeze(), dx, dy, dz),
            trilinear(dsets['uy'][slices].squeeze(), dx, dy, dz),
            trilinear(dsets['uz'][slices].squeeze(), dx, dy, dz)
        ])

    def close(self):
        for f in self.files: f.close()

class GliderPhysics:
    def __init__(self, polar_file_base, mass=2, area=0.3):
        self.m, self.A, self.g, self.rho = mass, area, 9.81, 1.225
        self.aero_interp = self.load_polar_data(polar_file_base)

    def load_polar_data(self, case_name):
        polar_name = case_name + ".polar"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, polar_name)
        df = pd.read_csv(full_path, sep='\s+')
        return {
            "Cl": interp1d(df['AoA'], df['CL'], kind='linear', fill_value="extrapolate"),
            "Cd": interp1d(df['AoA'], df['CDtot'], kind='linear', fill_value="extrapolate")
        }

    def get_steady_state(self, alpha_rad, bank_rad):
        cl = float(self.aero_interp['Cl'](np.degrees(alpha_rad)))
        cd = float(self.aero_interp['Cd'](np.degrees(alpha_rad)))
        
        # 1. 计算平衡下滑角 gamma: tan(gamma) = CD / (CL * cos(mu))
        tan_gamma = cd / (cl * np.cos(bank_rad))
        gamma_rad = np.arctan(tan_gamma)
        
        # 2. 计算平衡空速 v: v^2 = (2mg sin(gamma)) / (rho * S * CD)
        v_sq = (2 * self.m * self.g * np.sin(gamma_rad)) / (self.rho * self.A * cd)
        v_tas = np.sqrt(max(v_sq, 0.1))
        
        # 3. 计算航向变化率
        y_acc = self.g*np.cos(gamma_rad) * np.tan(bank_rad)
        dchi_dt = y_acc / v_tas / np.cos(gamma_rad)
        
        return v_tas, gamma_rad, dchi_dt

class GliderEnv(gym.Env):
    def __init__(self, h5_file_path, polar_file_base, control_mode=0, domain_size=(1000.0, 1000.0, 1000.0), 
                 dt_rl=1.0, n_phys_per_rl=2, rl_steps_per_frame=2, wind_ampf=12, hysteresis_pct=0.1, random_init = False):
        super().__init__()
        self.mode = control_mode
        self.wind_manager = RBWindField(h5_file_path, domain_size=domain_size)
        self.physics = GliderPhysics(polar_file_base)
        self.domain_size = np.array(domain_size)
        self.random_init = random_init
        
        # 时间控制参数
        self.dt_rl = dt_rl                             # RL步长 (秒)
        self.n_phys_per_rl = n_phys_per_rl             # 每个RL step内的物理积分步数
        self.rl_steps_per_frame = rl_steps_per_frame   # 多少个RL step更新一次风场数据帧
        self.dt_integration = dt_rl / n_phys_per_rl    # 每次物理积分的实际Delta T
        
        self.wind_ampf = wind_ampf
        self.b = 2.0  # 翼展
        self.reward_survive = 0
        self.rl_step_counter = 0                       # 用于追踪RL步数以更新风场

        # 状态空间离散化阈值
        self.bins_w_accel = np.array([-0.3, 0.3])
        self.bins_delta_w = np.array([-0.06, 0.06])

        # 动作与观测空间保持不变
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.MultiDiscrete([3, 3])

        bank_incre = np.deg2rad(15)
        aoa_incre = np.deg2rad(3)
        aoa_base = np.deg2rad(1)

        self._action_mapping = {
            0: (aoa_base,  -bank_incre),            1: (aoa_base,  0.0),            2: (aoa_base,            bank_incre),
            3: (aoa_base+aoa_incre, -bank_incre),   4: (aoa_base+aoa_incre, 0.0),   5: (aoa_base+aoa_incre,  bank_incre),
            6: (aoa_base+2*aoa_incre, -bank_incre), 7: (aoa_base+2*aoa_incre, 0.0), 8: (aoa_base+2*aoa_incre,bank_incre),
        }

        aoa_step = np.deg2rad(3)
        bank_step = np.deg2rad(5)
        self.aoa_step = aoa_step
        self.bank_step = bank_step

        self._inc_action_mapping = {
            0: (-1, -1), 1: (-1,  0), 2: (-1,  1),
            3: ( 0, -1), 4: ( 0,  0), 5: ( 0,  1),
            6: ( 1, -1), 7: ( 1,  0), 8: ( 1,  1)
        }
        
        # 增量模式的物理边界限制
        self.aoa_bounds = [np.deg2rad(0), np.deg2rad(12)]
        self.bank_bounds = [np.deg2rad(-20), np.deg2rad(20)]
        self.hysteresis_pct = hysteresis_pct
        
        # 用于记录上一次的分箱索引，实现施密特触发器逻辑
        self.last_idx_az = None
        self.last_idx_dw = None

    def step(self, action):
        if self.mode ==0:
            target_alpha, target_bank = self._action_mapping[action]
            self.control_state[0] = target_alpha
            self.control_state[1] = target_bank
        else:
            da_idx, db_idx = self._inc_action_mapping[action]
            self.control_state[0] += da_idx * self.aoa_step
            self.control_state[1] += db_idx * self.bank_step
            
            # 必须进行数值裁剪，防止超出极点图插值范围或发生物理翻转
            self.control_state[0] = np.clip(self.control_state[0], *self.aoa_bounds)
            self.control_state[1] = np.clip(self.control_state[1], *self.bank_bounds)

        sum_w_accel = 0.0
        sum_delta_w = 0.0
        terminated = False
        truncated = False

        # --- 物理积分循环 ---
        # 在这 n_phys_per_rl 步中，风场帧保持不变
        for _ in range(self.n_phys_per_rl):
            x, y, z, chi = self.phy_state
            
            # 获取当前位置风速 (使用当前锁定的风场帧)
            w_vec_start = self.wind_manager.get_wind(x, y, z) * self.wind_ampf
            v_tas, gamma, dchi_dt = self.physics.get_steady_state(self.control_state[0], self.control_state[1])
            
            # 位移计算使用自定义的 dt_integration
            dx = (v_tas * np.cos(gamma) * np.cos(chi) + w_vec_start[0]) * self.dt_integration
            dy = (v_tas * np.cos(gamma) * np.sin(chi) + w_vec_start[1]) * self.dt_integration
            dz = (-v_tas * np.sin(gamma) + w_vec_start[2]) * self.dt_integration
            
            self.phy_state[0] = (x + dx) % self.domain_size[0]
            self.phy_state[1] = (y + dy) % self.domain_size[1]
            self.phy_state[2] += dz
            self.phy_state[3] = (chi + dchi_dt * self.dt_integration) % (2 * np.pi)

            # 传感器测量：计算垂直风场加速度和翼尖风速差
            w_vec_end = self.wind_manager.get_wind(*self.phy_state[:3]) * self.wind_ampf
            sum_w_accel += (w_vec_end[2] - w_vec_start[2]) / self.dt_integration
            
            side_vec = np.array([np.sin(chi), -np.cos(chi), 0])
            pos_r = self.phy_state[:3] + (self.b / 2.0) * side_vec
            pos_r[:2] %= self.domain_size[:2]
            pos_r[2] = np.clip(pos_r[2], 0, self.domain_size[2] - 0.01)
            pos_l = (self.phy_state[:3] - (self.b / 2.0) * side_vec) % self.domain_size
            w_r = self.wind_manager.get_wind(*pos_r)[2] * self.wind_ampf
            w_l = self.wind_manager.get_wind(*pos_l)[2] * self.wind_ampf
            sum_delta_w += (w_r - w_l)

            if (self.phy_state[2] <= self.domain_size[2] * 0.1) or (self.phy_state[2] >= self.domain_size[2] * 0.9):
                terminated = True
                break

        # --- RL步数管理与风场帧更新 ---
        self.rl_step_counter += 1
        if self.rl_step_counter % self.rl_steps_per_frame == 0:
            # 只有达到指定步数才切换到下一个 H5 数据帧
            if not self.wind_manager.step_time():
                truncated = True  # 数据读完了

        # 计算本步平均传感器数值
        self.w_accel = sum_w_accel / self.n_phys_per_rl
        self.delta_w = sum_delta_w / self.n_phys_per_rl
        
        # 奖励计算
        current_uz = self.wind_manager.get_wind(*self.phy_state[:3])[2] * self.wind_ampf
        reward = current_uz + 5 * self.w_accel + self.reward_survive

        if terminated or truncated:
            reward += (self.phy_state[2]-self.initial_z)*0.2

        info = {
            "w_accel": self.w_accel, 
            "delta_w": self.delta_w, 
            "control": self.control_state, 
            "height": self.phy_state[2],
            "tas": v_tas,
            "uz": current_uz
        }
        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rl_step_counter = 0  # 重置计数器
        self.wind_manager.reset(options.get("resettime", 0) if options else 0)
        if self.random_init:
            x, y = self.np_random.uniform(0.2, 0.8, size=2) * self.domain_size[:2]
            z = self.np_random.uniform(0.2, 0.6) * self.domain_size[2]
        else:
            x,y,z = 0.5 * self.domain_size
        self.initial_z = z
        self.phy_state = np.array([x, y, z, self.np_random.uniform(0, 2*np.pi)])
        self.control_state = np.array([np.deg2rad(5.0), 0.0])
        self.w_accel, self.delta_w = 0.0, 0.0
        self.last_idx_az = None
        self.last_idx_dw = None

        return self._get_obs(), {"height": self.phy_state[2]}
    
    def _apply_hysteresis(self, value, bins, last_idx):
        # 初始步，直接返回 digitize 结果
        if last_idx is None:
            return np.digitize(value, bins)

        # 获取自然分箱位置
        new_idx = np.digitize(value, bins)
        
        # 如果分箱没变，直接返回
        if new_idx == last_idx:
            return last_idx

        target_bin_idx = last_idx if new_idx > last_idx else new_idx
        threshold = bins[target_bin_idx]
        margin = abs(threshold) * self.hysteresis_pct

        if new_idx > last_idx:
            return new_idx if value > threshold + margin else last_idx
        else:
            return new_idx if value < threshold - margin else last_idx

    def _get_obs(self):
        # 分别计算带迟滞的索引
        self.last_idx_az = self._apply_hysteresis(self.w_accel, self.bins_w_accel, self.last_idx_az)
        self.last_idx_dw = self._apply_hysteresis(self.delta_w, self.bins_delta_w, self.last_idx_dw)
        
        return np.array([self.last_idx_az, self.last_idx_dw], dtype=np.int32)