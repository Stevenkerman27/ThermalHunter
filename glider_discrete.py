import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from scipy.interpolate import interp1d
import h5py
import os
import re

class RBWindField:
    def __init__(self, h5_paths, domain_size=(1000, 1000, 1000)):
        # 定义自然排序函数
        def natural_key(string_):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

        if isinstance(h5_paths, list):
            # 使用自定义 key 进行排序，确保 s2 排在 s10 前面
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
        self.dt_phy = 0.0  # 物理时间步长

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
        
        # 自动计算物理步长 (假设等间距)
        if len(self.all_sim_times) > 1:
            self.dt_phy = self.all_sim_times[1] - self.all_sim_times[0]
        
        first_shape = self.dsets_list[0]['ux'].shape
        self.space_range[0] = first_shape[self.x_axis]
        self.space_range[1] = first_shape[self.y_axis]
        self.space_range[2] = first_shape[self.z_axis]
        
        print(f"WindField initialized. dt_phy: {self.dt_phy:.4f}, Total steps: {self.max_t_idx + 1}")

    def reset(self, t_index=None):
        if t_index is None:
            self.global_t_idx = np.random.randint(0, max(1, self.max_t_idx - 1))
        else:
            self.global_t_idx = min(t_index, self.max_t_idx)
        self._update_file_pointers()
        return self.global_t_idx

    def _update_file_pointers(self):
        """根据当前的 global_t_idx 更新文件和局部索引"""
        for i in range(len(self.t_offsets) - 1):
            if self.t_offsets[i] <= self.global_t_idx < self.t_offsets[i+1]:
                self.current_file_idx = i
                self.local_t_idx = self.global_t_idx - self.t_offsets[i]
                break

    def step_time(self):
        """推进一个物理时间步"""
        if self.global_t_idx < self.max_t_idx:
            self.global_t_idx += 1
            self._update_file_pointers()
            return True
        return False # 已到达数据末尾

    def get_wind(self, x, y, z):
        # 归一化映射逻辑 (保持不变)
        fx = np.clip((x / self.domain_size[0]) * (self.space_range[0] - 1), 0, self.space_range[0] - 1.00001)
        fy = np.clip((y / self.domain_size[1]) * (self.space_range[1] - 1), 0, self.space_range[1] - 1.00001)
        fz = np.clip((z / self.domain_size[2]) * (self.space_range[2] - 1), 0, self.space_range[2] - 1.00001)

        ix0, iy0, iz0 = int(fx), int(fy), int(fz)
        dx, dy, dz = fx - ix0, fy - iy0, fz - iz0

        slices = (slice(self.local_t_idx, self.local_t_idx + 1), slice(ix0, ix0 + 2), slice(iy0, iy0 + 2), slice(iz0, iz0 + 2))
        dsets = self.dsets_list[self.current_file_idx]
        
        def trilinear(cube, dx, dy, dz):
            c0 = cube[0] * (1 - dx) + cube[1] * dx
            c1 = c0[0] * (1 - dy) + c0[1] * dy
            return c1[0] * (1 - dz) + c1[1] * dz

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

    @staticmethod
    def load_polar_data(case_name):
        polar_name = case_name + ".polar"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, polar_name)
        df = pd.read_csv(full_path, sep='\s+')
        return {
            "Cl": interp1d(df['AoA'], df['CL'], kind='linear', fill_value="extrapolate"),
            "Cd": interp1d(df['AoA'], df['CDtot'], kind='linear', fill_value="extrapolate")
        }

    def get_forces_and_derivatives(self, state, control, wind_vec, wind_accel_vec, V_min=5, damping=0.4):
        x, y, z, V_tas, gamma, chi = state
        alpha_rad, phi_rad = control
        
        V_tas_eff = max(V_tas, 1.0)
        cl = self.aero_interp['Cl'](np.degrees(alpha_rad))
        cd = self.aero_interp['Cd'](np.degrees(alpha_rad))
        q_bar = 0.5 * self.rho * (V_tas_eff ** 2)
        
        L = q_bar * self.A * cl
        D = q_bar * self.A * cd

        if V_tas < V_min:
            L *= 0.2 # 模拟失速

        # wind_accel_vec 是地速坐标系下的风加速度
        dv_dt = -D / self.m - self.g * np.sin(gamma) - (
            wind_accel_vec[0] * np.cos(gamma) * np.cos(chi) +
            wind_accel_vec[1] * np.cos(gamma) * np.sin(chi) +
            wind_accel_vec[2] * np.sin(gamma)
        )

        dgamma_dt = (L / self.m * np.cos(phi_rad) - self.g * np.cos(gamma)) / V_tas_eff
        dchi_dt = (L / self.m * np.sin(phi_rad)) / (V_tas_eff * np.cos(gamma) + 1e-6)
        
        # 俯仰阻尼
        gamma_damping = damping * dgamma_dt
        dgamma_dt -= gamma_damping
        
        dx_dt = V_tas * np.cos(gamma) * np.cos(chi) + wind_vec[0]
        dy_dt = V_tas * np.cos(gamma) * np.sin(chi) + wind_vec[1]
        dz_dt = V_tas * np.sin(gamma) + wind_vec[2]
    
        return np.array([dx_dt, dy_dt, dz_dt, dv_dt, dgamma_dt, dchi_dt])

    def integration_step(self, state, control, wind_manager, wind_ampf, dt):
        # 获取当前风速
        w0 = wind_manager.get_wind(*state[0:3]) * wind_ampf
        
        # 先进行一次欧拉预测，估算下一时刻位置的风速
        # 粗略估计地速导数（不考虑风梯度）用于计算下一位置
        k1_simple = self.get_forces_and_derivatives(state, control, w0, np.zeros(3))
        next_pos_est = state[0:3] + k1_simple[0:3] * dt
        
        # 获取预测位置的风速并计算风加速度 dW/dt
        w1 = wind_manager.get_wind(*next_pos_est) * wind_ampf
        wind_accel_vec = (w1 - w0) / dt
        
        # 使用修正后的导数进行正式更新
        derivatives = self.get_forces_and_derivatives(state, control, w0, wind_accel_vec)
        new_state = state + derivatives * dt
        
        if new_state[3] < 2: 
            new_state[3] = 2
        return new_state

class GliderEnv(gym.Env):
    def __init__(self, h5_file_path, polar_file_base, domain_size=(1000.0, 1000.0, 1000.0), n_steps_per_rl=4, wind_ampf = 100, V_ini = 12):
        super().__init__()
        self.wind_manager = RBWindField(h5_file_path, domain_size=domain_size)
        self.physics = GliderPhysics(polar_file_base)
        self.domain_size = domain_size
        
        # RL 步长设置
        self.n_steps_per_rl = n_steps_per_rl
        self.dt_phy = self.wind_manager.dt_phy
        self.dt_rl = self.dt_phy * self.n_steps_per_rl
        # 风速倍数
        self.wind_ampf = wind_ampf
        self.V_ini = V_ini

        self.b = 2 #翼展

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self._action_to_move = {
            0: (np.deg2rad(2.5), np.deg2rad(5.0)),  1: (np.deg2rad(2.5), 0.0), 2: (np.deg2rad(2.5), np.deg2rad(-5.0)),
            3: (0.0, np.deg2rad(5.0)),             4: (0.0, 0.0),             5: (0.0, np.deg2rad(-5.0)),
            6: (np.deg2rad(-2.5), np.deg2rad(5.0)), 7: (np.deg2rad(-2.5), 0.0), 8: (np.deg2rad(-2.5), np.deg2rad(-5.0)),
        }

    def step(self, action):
        # 1. 计算开始前的单位质量机械能 (Specific Mechanical Energy)
        v_start = self.phy_state[3]
        z_start = self.phy_state[2]
        energy_start = 0.5 * (v_start**2) + self.physics.g * z_start
        
        # 执行动作更新 (原有逻辑)
        d_alpha, d_bank = self._action_to_move[action]
        cur_alpha, cur_phi = self.control_state
        self.control_state = np.array([
            np.clip(cur_alpha + d_alpha, np.deg2rad(-5), np.deg2rad(15)),
            np.clip(cur_phi + d_bank, np.deg2rad(-45), np.deg2rad(45))
        ])

        truncated = False
        for _ in range(self.n_steps_per_rl):
            self.phy_state = self.physics.integration_step(
                self.phy_state, self.control_state, self.wind_manager, self.wind_ampf, self.dt_phy
            )
            if not self.wind_manager.step_time():
                truncated = True
                break
            if self.phy_state[2] <= 0: break

        # 2. 计算 delta_w (左右翼尖风速差)
        x, y, z, V, gamma, chi = self.phy_state
        # 计算指向右侧的单位向量 (垂直于航向 chi)
        side_vec = np.array([np.sin(chi), -np.cos(chi), 0])
        pos_right = self.phy_state[0:3] + (self.b / 2.0) * side_vec
        pos_left  = self.phy_state[0:3] - (self.b / 2.0) * side_vec
        
        w_right = self.wind_manager.get_wind(*pos_right)[2] * self.wind_ampf
        w_left  = self.wind_manager.get_wind(*pos_left)[2] * self.wind_ampf
        self.delta_w = w_right - w_left # 记录到 self 以便 obs 使用

        # 3. 计算 w_accel (垂直风加速度)
        current_wind_z = self.wind_manager.get_wind(*self.phy_state[0:3])[2] * self.wind_ampf
        self.w_accel = (current_wind_z - self.prev_w) / self.dt_rl
        self.prev_w = current_wind_z

        # 4. 计算结束后的能量及奖励
        energy_end = 0.5 * (self.phy_state[3]**2) + self.physics.g * self.phy_state[2]
        reward = energy_end - energy_start

        obs = self._get_obs()
        terminated = (z <= 0 or x < 0 or x > self.domain_size[0] or y < 0 or y > self.domain_size[1])
        
        if terminated and z <= 0:
            reward -= 100.0 

        # 将这两个值放入 info 字典，方便 simulator.py 统计
        info = {
            "w_accel": self.w_accel,
            "delta_w": self.delta_w
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.wind_manager.reset()
        x, y = self.np_random.uniform(0.4, 0.6, size=2) * self.domain_size[:2]
        z = 0.8 * self.domain_size[2]
        self.phy_state = np.array([x, y, z, self.V_ini, 0.0, self.np_random.uniform(0, 2*np.pi)], dtype=np.float32)
        self.control_state = np.array([0.0, 0.0], dtype=np.float32)

        # 获取初始位置的垂直风速
        initial_wind = self.wind_manager.get_wind(*self.phy_state[0:3])
        self.prev_w = initial_wind[2] * self.wind_ampf  # 记录初始垂直风速
        self.w_accel = 0.0  # 初始化加速度为 0
        return self._get_obs(), {}

    def _get_obs(self):
        x, y, z, V, gamma, chi = self.phy_state
        alpha, phi = self.control_state
        
        return np.array([V/self.V_ini,alpha,self.w_accel,self.delta_w],dtype=np.float32)

    def close(self):
        self.wind_manager.close()