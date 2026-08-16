import numpy as np
import gymnasium as gym
from gymnasium import spaces
import h5py
import re
import os
import pandas as pd
from scipy.interpolate import interp1d
from numba import njit
import config

@njit
def trilinear(cube, dx, dy, dz):
    c0 = cube[0] * (1 - dx) + cube[1] * dx
    c1 = c0[0] * (1 - dy) + c0[1] * dy
    return c1[0] * (1 - dz) + c1[1] * dz

class RBWindField:
    def __init__(self, h5_paths, domain_size=config.DOMAIN_SIZE, memory_mode=True):
        if isinstance(h5_paths, list):
            self.h5_paths = sorted(h5_paths, key=config.natural_key) 
        else:
            self.h5_paths = [h5_paths]

        self.domain_size = np.array(domain_size, dtype=np.float32)
        self.memory_mode = memory_mode
        
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
            
            if self.memory_mode:
                # 直接将整个数据集读取到内存中，存储为字典形式的 numpy array
                dset_group = {
                    'ux': f['tasks/ux'][:],
                    'uy': f['tasks/uy'][:],
                    'uz': f['tasks/uz'][:],
                    'buoyancy': f['tasks/buoyancy'][:]
                }
            else:
                # 仅保留数据集句柄，延迟读取
                dset_group = {
                    'ux': f['tasks/ux'],
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
        self.space_range[0] = first_shape[self.x_axis]
        self.space_range[1] = first_shape[self.y_axis]
        self.space_range[2] = first_shape[self.z_axis]
        
        mode_str = "Memory" if self.memory_mode else "Disk (Lazy)"
        print(f"WindField initialized ({mode_str} mode). dt_phy: {self.dt_phy:.4f}, Total steps: {self.max_t_idx + 1}")

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
        return self._get_wind_at_index(self.global_t_idx, x, y, z)

    def _get_wind_at_index(self, global_t_idx, x, y, z):
        global_t_idx = int(np.clip(global_t_idx, 0, self.max_t_idx))
        file_idx = int(np.searchsorted(self.t_offsets, global_t_idx, side='right') - 1)
        file_idx = min(file_idx, len(self.dsets_list) - 1)
        local_t_idx = global_t_idx - self.t_offsets[file_idx]
        fx = np.clip((x / self.domain_size[0]) * (self.space_range[0] - 1), 0, self.space_range[0] - 1.00001)
        fy = np.clip((y / self.domain_size[1]) * (self.space_range[1] - 1), 0, self.space_range[1] - 1.00001)
        fz = np.clip((z / self.domain_size[2]) * (self.space_range[2] - 1), 0, self.space_range[2] - 1.00001)

        ix0, iy0, iz0 = int(fx), int(fy), int(fz)
        dx, dy, dz = fx - ix0, fy - iy0, fz - iz0

        slices = (slice(local_t_idx, local_t_idx + 1), slice(ix0, ix0 + 2), slice(iy0, iy0 + 2), slice(iz0, iz0 + 2))
        dsets = self.dsets_list[file_idx]

        return np.array([
            trilinear(dsets['ux'][slices].squeeze(), dx, dy, dz),
            trilinear(dsets['uy'][slices].squeeze(), dx, dy, dz),
            trilinear(dsets['uz'][slices].squeeze(), dx, dy, dz)
        ])

    def get_wind_at_frame(self, frame_position, x, y, z):
        """Linearly interpolate wind between adjacent global data frames."""
        bounded_frame = float(np.clip(frame_position, 0.0, self.max_t_idx))
        lower_idx = int(np.floor(bounded_frame))
        upper_idx = min(lower_idx + 1, self.max_t_idx)
        fraction = bounded_frame - lower_idx
        lower_wind = self._get_wind_at_index(lower_idx, x, y, z)
        if upper_idx == lower_idx:
            return lower_wind
        upper_wind = self._get_wind_at_index(upper_idx, x, y, z)
        return lower_wind + fraction * (upper_wind - lower_wind)

    def vector_rms(self):
        total_squared_speed = 0.0
        component_count = 0
        for dsets in self.dsets_list:
            for component in ('ux', 'uy', 'uz'):
                values = dsets[component][:]
                total_squared_speed += np.square(values, dtype=np.float64).sum()
                component_count += values.size
        if component_count == 0:
            raise ValueError("wind field contains no velocity samples")
        return float(np.sqrt(total_squared_speed / component_count))

    def close(self):
        for f in self.files:
            f.close()
        self.dsets_list.clear()
        import gc
        gc.collect()

class GliderPhysics:
    def __init__(self, polar_file_base, mass=config.MASS, area=config.AREA):
        self.m, self.A, self.g, self.rho = mass, area, 9.81, 1.225
        self.aero_interp = self.load_polar_data(polar_file_base)

    def load_polar_data(self, case_name):
        polar_name = case_name + ".polar"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, polar_name)
        df = pd.read_csv(full_path, sep=r'\s+')
        return {
            "Cl": interp1d(df['AoA'], df['CL'], kind='linear', fill_value="extrapolate"),
            "Cd": interp1d(df['AoA'], df['CDtot'], kind='linear', fill_value="extrapolate")
        }

    def get_steady_state(self, alpha_rad, bank_rad, drag_mult=1.0):
        cl = float(self.aero_interp['Cl'](np.degrees(alpha_rad)))
        cd = float(self.aero_interp['Cd'](np.degrees(alpha_rad))) * drag_mult

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


def compute_wind_amplification(wind_manager, physics):
    median_aoa_deg = config.AOA_MIN_DEG + (config.AOA_BINS // 2) * config.AOA_STEP_DEG
    typical_tas, _, _ = physics.get_steady_state(np.deg2rad(median_aoa_deg), 0.0)
    wind_rms = wind_manager.vector_rms()
    if wind_rms <= 0.0:
        raise ValueError("wind field RMS must be positive")
    return config.WIND_RMS_TO_TAS_RATIO * typical_tas / wind_rms

class GliderEnv(gym.Env):
    # --- Centralized RL Configuration (From config.py) ---
    BANK_MIN_DEG = config.BANK_MIN_DEG
    BANK_MAX_DEG = config.BANK_MAX_DEG
    BANK_STEP_DEG = config.BANK_STEP_DEG
    BANK_BINS = config.BANK_BINS
    
    AOA_MIN_DEG = config.AOA_MIN_DEG
    AOA_MAX_DEG = config.AOA_MAX_DEG
    AOA_STEP_DEG = config.AOA_STEP_DEG
    AOA_BINS = config.AOA_BINS
    
    BINS_W_ACCEL = config.BINS_W_ACCEL
    BINS_DELTA_W = config.BINS_DELTA_W
    
    # String labels for external visualization scripts
    ACTION_LABELS = {
        0: "A-B-", 1: "A-B0", 2: "A-B+",
        3: "A0B-", 4: "A0B0", 5: "A0B+",
        6: "A+B-", 7: "A+B0", 8: "A+B+"
    }
    OBS_WIND_SYMBOLS = ["-", "0", "+"]

    def __init__(self, h5_file_path, polar_file_base, domain_size=config.DOMAIN_SIZE, 
                 dt_rl=config.DT_RL, n_phys_per_rl=config.N_PHYS_PER_RL, rl_steps_per_frame=config.RL_STEPS_PER_FRAME, 
                 wind_ampf=None, hysteresis_pct=config.HYSTERESIS_PCT, random_init=True,
                 reward_w_accel=config.REWARD_W_ACCEL_WEIGHT, memory_mode=False, continuous_obs=False):
        super().__init__()
        self.wind_manager = RBWindField(h5_file_path, domain_size=domain_size, memory_mode=memory_mode)
        self.physics = GliderPhysics(polar_file_base)
        self.domain_size = np.array(domain_size)
        self.random_init = random_init
        self.continuous_obs = continuous_obs
        
        # 时间控制参数
        self.dt_rl = dt_rl                             # RL步长 (秒)
        self.n_phys_per_rl = n_phys_per_rl             # 每个RL step内的物理积分步数
        self.rl_steps_per_frame = rl_steps_per_frame   # 多少个RL step更新一次风场数据帧
        self.dt_integration = dt_rl / n_phys_per_rl    # 每次物理积分的实际Delta T
        
        self.wind_ampf = compute_wind_amplification(self.wind_manager, self.physics) if wind_ampf is None else wind_ampf
        print(f"Wind amplification: {self.wind_ampf:.6f} (target RMS/TAS={config.WIND_RMS_TO_TAS_RATIO:.3f})")
        self.b = config.WINGSPAN
        self.reward_w_accel = reward_w_accel
        self.rl_step_counter = 0                       # 用于追踪RL步数以更新风场

        # 动作与观测空间
        # Action space: 9 actions (3x3 grid for AoA and Bank deltas)
        self.action_space = spaces.Discrete(9)
        # Observation space
        if self.continuous_obs:
            # [aoa_idx, bank_idx, w_accel, delta_w]
            # Use +/- 10.0 for sensor values as broad limits; DQN handles raw values better with normalization anyway
            low = np.array([0, 0, -10.0, -10.0], dtype=np.float32)
            high = np.array([self.AOA_BINS - 1, self.BANK_BINS - 1, 10.0, 10.0], dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self.observation_space = spaces.MultiDiscrete([self.AOA_BINS, self.BANK_BINS, 3, 3])

        self.hysteresis_pct = hysteresis_pct
        # 用于记录上一次的分箱索引，实现施密特触发器逻辑
        self.last_idx_az = None
        self.last_idx_dw = None

        # --- 方案 1: 物理预计算 ---
        self._precompute_physics()

    def _precompute_physics(self):
        """预先计算所有离散 AoA 和 bank 角度下的平衡物理状态 (包括正常与高阻力状态)"""
        self.physics_table = np.zeros((self.AOA_BINS, self.BANK_BINS, 3), dtype=np.float32)
        self.physics_table_drag = np.zeros((self.AOA_BINS, self.BANK_BINS, 3), dtype=np.float32)
        
        for a_idx in range(self.AOA_BINS):
            aoa_rad = np.deg2rad(self.AOA_MIN_DEG + a_idx * self.AOA_STEP_DEG)
            for b_idx in range(self.BANK_BINS):
                bank_rad = np.deg2rad(self.BANK_MIN_DEG + b_idx * self.BANK_STEP_DEG)
                
                # 正常状态
                v_tas, gamma, dchi_dt = self.physics.get_steady_state(aoa_rad, bank_rad)
                self.physics_table[a_idx, b_idx] = [v_tas, gamma, dchi_dt]
                
                # 操纵面产生的额外阻力状态
                v_tas_d, gamma_d, dchi_dt_d = self.physics.get_steady_state(aoa_rad, bank_rad, drag_mult=config.CONTROL_DRAG_MULTIPLIER)
                self.physics_table_drag[a_idx, b_idx] = [v_tas_d, gamma_d, dchi_dt_d]
        
        print(f"Physics tables pre-computed (Normal & Drag-Penalty) for {self.AOA_BINS} AoA and {self.BANK_BINS} bank angles.")

    def step(self, action):
        # Decode action into deltas: 0-8 maps to (aoa_delta, bank_delta) in [-1, 0, 1]^2
        aoa_delta = (action // 3) - 1
        bank_delta = (action % 3) - 1
        
        # 记录是否发生了控制改变
        control_changed = (aoa_delta != 0) or (bank_delta != 0)

        self.aoa_idx = np.clip(self.aoa_idx + aoa_delta, 0, self.AOA_BINS - 1)
        self.bank_idx = np.clip(self.bank_idx + bank_delta, 0, self.BANK_BINS - 1)
        
        # 从预计算表中获取参数
        if control_changed:
            v_tas, gamma, dchi_dt = self.physics_table_drag[self.aoa_idx, self.bank_idx]
        else:
            v_tas, gamma, dchi_dt = self.physics_table[self.aoa_idx, self.bank_idx]
        
        # 仅用于 info 字典的记录
        aoa_rad = np.deg2rad(self.AOA_MIN_DEG + self.aoa_idx * self.AOA_STEP_DEG)
        bank_rad = np.deg2rad(self.BANK_MIN_DEG + self.bank_idx * self.BANK_STEP_DEG)

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
            
            # 位移计算使用预计算出的 v_tas, gamma, dchi_dt
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
            
            pos_l = self.phy_state[:3] - (self.b / 2.0) * side_vec
            pos_l[:2] %= self.domain_size[:2]
            pos_l[2] = np.clip(pos_l[2], 0, self.domain_size[2] - 0.01)
            
            w_r = self.wind_manager.get_wind(*pos_r)[2] * self.wind_ampf
            w_l = self.wind_manager.get_wind(*pos_l)[2] * self.wind_ampf
            sum_delta_w += (w_r - w_l)

            if (self.phy_state[2] <= self.domain_size[2] * 0.1) or (self.phy_state[2] >= self.domain_size[2] * 0.9):
                terminated = True
                self.last_terminal_height = self.phy_state[2]
                break

        # --- RL步数管理与风场帧更新 ---
        self.rl_step_counter += 1
        if self.rl_step_counter % self.rl_steps_per_frame == 0:
            # 只有达到指定步数才切换到下一个 H5 数据帧
            if not self.wind_manager.step_time():
                truncated = True  # 数据读完了
                self.last_terminal_height = self.phy_state[2]

        # 计算本步平均传感器数值
        self.w_accel = sum_w_accel / self.n_phys_per_rl
        self.delta_w = sum_delta_w / self.n_phys_per_rl
        
        # 奖励计算
        current_uz = self.wind_manager.get_wind(*self.phy_state[:3])[2] * self.wind_ampf
        reward = current_uz + self.reward_w_accel * self.w_accel

        info = {
            "w_accel": self.w_accel, 
            "delta_w": self.delta_w, 
            "control": [aoa_rad, bank_rad], 
            "height": self.phy_state[2],
            "tas": v_tas,
            "uz": current_uz,
            "prev_height": getattr(self, "last_terminal_height", 0.0)
        }
        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Capture final height of previous episode
        prev_h = getattr(self, 'last_terminal_height', 0.0)

        self.rl_step_counter = 0  # 重置计数器
        self.wind_manager.reset(options.get("resettime", 0) if options else 0)
        if self.random_init:
            x, y = self.np_random.uniform(0.2, 0.8, size=2) * self.domain_size[:2]
            z = self.np_random.uniform(0.2, 0.6) * self.domain_size[2]
            init_dir = self.np_random.uniform(0, 2*np.pi)
        else:
            x,y,z = 0.5 * self.domain_size
            init_dir = 0
        self.initial_z = z
        self.phy_state = np.array([x, y, z, init_dir])
        self.aoa_idx = self.AOA_BINS // 2   # Start at middle AoA
        self.bank_idx = self.BANK_BINS // 2 # Start at 0 degrees bank
        self.w_accel, self.delta_w = 0.0, 0.0
        self.last_idx_az = None
        self.last_idx_dw = None

        return self._get_obs(), {"height": self.phy_state[2], "prev_height": prev_h}
    
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
        if self.continuous_obs:
            return np.array([self.aoa_idx, self.bank_idx, self.w_accel, self.delta_w], dtype=np.float32)

        # 分别计算带迟滞的索引
        self.last_idx_az = self._apply_hysteresis(self.w_accel, self.BINS_W_ACCEL, self.last_idx_az)
        self.last_idx_dw = self._apply_hysteresis(self.delta_w, self.BINS_DELTA_W, self.last_idx_dw)
        
        return np.array([self.aoa_idx, self.bank_idx, self.last_idx_az, self.last_idx_dw], dtype=np.int32)
    
    def close(self):
        if hasattr(self, 'wind_manager'):
            self.wind_manager.close()
