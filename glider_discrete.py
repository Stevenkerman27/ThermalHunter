import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from scipy.interpolate import interp1d
import h5py
import os

class RBWindField:
    def __init__(self, h5_paths, domain_size=(1000, 1000, 1000)):
        # 支持传入列表或单个路径
        self.h5_paths = sorted(h5_paths) if isinstance(h5_paths, list) else [h5_paths]
        self.domain_size = np.array(domain_size, dtype=np.float32)
        
        self.files = []
        self.dsets_list = [] # 存储每个文件的 dataset 引用
        self.t_offsets = [0] # 时间步偏移映射
        
        # 轴索引定义
        self.t_axis, self.x_axis, self.y_axis, self.z_axis = 0, 1, 2, 3
        self.space_range = [0, 0, 0]

        self.global_t_idx = 0
        self.current_file_idx = 0
        self.local_t_idx = 0
        self.all_sim_times = [] # 新增：存储全局物理时间序列

        self._open_resources()

    def _open_resources(self):
        for path in self.h5_paths:
            f = h5py.File(path, 'r')
            self.files.append(f)
            dset_group = {k: f['tasks'][k] for k in ['ux', 'uy', 'uz']}
            self.dsets_list.append(dset_group)
            
            # --- 新增：读取并累加物理时间 ---
            # 从任意一个 task 中提取 sim_time 标尺
            file_times = dset_group['ux'].dims[0]['sim_time'][:]
            self.all_sim_times.extend(file_times)
            
            file_t_steps = len(file_times)
            self.t_offsets.append(self.t_offsets[-1] + file_t_steps)
        
        self.all_sim_times = np.array(self.all_sim_times)
        self.max_t_idx = len(self.all_sim_times) - 1
        
        # 使用第一个文件初始化空间范围（假设所有 snapshot 空间网格一致）
        first_shape = self.dsets_list[0]['ux'].shape
        self.space_range[0] = first_shape[self.x_axis]
        self.space_range[1] = first_shape[self.y_axis]
        self.space_range[2] = first_shape[self.z_axis]
        
        print(f"WindField initialized with {len(self.files)} files.")
        print(f"Total time steps: {self.max_t_idx + 1}, Space: {self.space_range}")
        
    def get_current_time(self):
        """返回当前全局索引对应的物理时间"""
        return self.all_sim_times[self.global_t_idx]

    def reset(self, t_index=None):
        if t_index is None:
            # 随机选择时留出 buffer，防止越界
            self.global_t_idx = np.random.randint(0, max(1, self.max_t_idx - 1))
        else:
            self.global_t_idx = min(t_index, self.max_t_idx)
        
        # 确定当前全局索引落在哪个文件里
        for i in range(len(self.t_offsets) - 1):
            if self.t_offsets[i] <= self.global_t_idx < self.t_offsets[i+1]:
                self.current_file_idx = i
                self.local_t_idx = self.global_t_idx - self.t_offsets[i]
                break
        return self.global_t_idx

    def get_wind(self, x, y, z):
        # 1. 映射坐标
        fx = np.clip(x * self.space_range[0], 0, self.space_range[0] - 1.00001)
        fy = np.clip(y * self.space_range[1], 0, self.space_range[1] - 1.00001)
        fz = np.clip(z * self.space_range[2], 0, self.space_range[2] - 1.00001)

        ix0, iy0, iz0 = int(fx), int(fy), int(fz)
        dx, dy, dz = fx - ix0, fy - iy0, fz - iz0

        # 2. 从当前活跃的文件中切片
        slices = (
            slice(self.local_t_idx, self.local_t_idx + 1),
            slice(ix0, ix0 + 2),
            slice(iy0, iy0 + 2),
            slice(iz0, iz0 + 2)
        )

        dsets = self.dsets_list[self.current_file_idx]
        u_block = dsets['ux'][slices].squeeze()
        v_block = dsets['uy'][slices].squeeze()
        w_block = dsets['uz'][slices].squeeze()

        # 3. 三线性插值 (保持原逻辑)
        def trilinear_simplified(cube, dx, dy, dz):
            c0 = cube[0] * (1 - dx) + cube[1] * dx
            c1 = c0[0] * (1 - dy) + c0[1] * dy
            return c1[0] * (1 - dz) + c1[1] * dz

        return np.array([
            trilinear_simplified(u_block, dx, dy, dz),
            trilinear_simplified(v_block, dx, dy, dz),
            trilinear_simplified(w_block, dx, dy, dz)
        ])
    
    def close(self):
        for f in self.files:
            f.close()
        self.files = []


# ==========================================
# 2. 物理引擎 (Physics Engine)
# ==========================================

class GliderPhysics:
    def __init__(self, polar_file_base, mass=2, area=0.3):
        """
        :param polar_file_base: 气动极曲线文件名前缀 (不含 _DegenGeom.polar)
        """
        self.m = mass
        self.A = area
        self.g = 9.81
        self.rho = 1.225
        
        # 集成气动数据读取
        self.aero_interp = self.load_polar_data(polar_file_base)

    @staticmethod
    def load_polar_data(case_name):
        """读取并处理气动数据 (静态方法)"""
        polar_name = case_name + ".polar"
        # 获取当前脚本所在的绝对路径 (RL-soar 文件夹路径)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, polar_name)

        print(f"[Info] 尝试加载气动文件: {full_path}") # 打印出来方便调试

        try:
            # 3. 修复 Pandas 警告: 使用 sep='\s+' 替代 delim_whitespace=True
            df = pd.read_csv(full_path, sep='\s+') 
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到气动文件: {full_path}")
        
        AoA = df['AoA'].to_numpy()
        Cl  = df['CL'].to_numpy()
        Cd  = df['CDtot'].to_numpy()
        CMy = df['CMy'].to_numpy()

        return {
            "Cl":  interp1d(AoA, Cl,  kind='linear', fill_value="extrapolate"),
            "Cd":  interp1d(AoA, Cd,  kind='linear', fill_value="extrapolate"),
            "CMy": interp1d(AoA, CMy, kind='linear', fill_value="extrapolate")
        }

    def get_forces_and_derivatives(self, state, control, wind_vec):
        x, y, z, V_tas, gamma, chi = state
        alpha_rad, phi_rad = control
        V_tas = max(V_tas, 0.1)

        # 气动计算
        alpha_deg = np.degrees(alpha_rad)
        cl = self.aero_interp['Cl'](alpha_deg)
        cd = self.aero_interp['Cd'](alpha_deg)
        
        q_bar = 0.5 * self.rho * (V_tas ** 2)
        L = q_bar * self.A * cl
        D = q_bar * self.A * cd
        
        # 动力学方程
        acc_aero_tangential = -D / self.m
        acc_aero_normal     =  L / self.m
        
        dv_dt = acc_aero_tangential - self.g * np.sin(gamma)
        dgamma_dt = (acc_aero_normal * np.cos(phi_rad) - self.g * np.cos(gamma)) / V_tas
        dchi_dt = (acc_aero_normal * np.sin(phi_rad)) / (V_tas * np.cos(gamma) + 1e-6)

        # 运动学 (加入风速)
        dx_dt = V_tas * np.cos(gamma) * np.cos(chi) + wind_vec[0]
        dy_dt = V_tas * np.cos(gamma) * np.sin(chi) + wind_vec[1]
        dz_dt = V_tas * np.sin(gamma)               + wind_vec[2]

        return np.array([dx_dt, dy_dt, dz_dt, dv_dt, dgamma_dt, dchi_dt])

    def integration_step(self, state, control, wind_vec, dt):
        derivs = self.get_forces_and_derivatives(state, control, wind_vec)
        new_state = state + derivs * dt
        return new_state


# ==========================================
# 3. RL 环境 (Gymnasium)
# ==========================================

class GliderEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 h5_file_path,
                 polar_file_base,
                 domain_size=(1.0, 1.0, 1.0), # 注意：这里要跟HDF5的实际物理尺度对应
                 alpha_step=2.5,
                 bank_step=5.0):
        
        super().__init__()
        
        # 1. 初始化物理与环境
        self.wind_manager = RBWindField(h5_file_path, domain_size=domain_size)
        self.physics = GliderPhysics(polar_file_base)
        self.domain_size = domain_size

        # 2. 空间定义
        self.alpha_step_rad = np.deg2rad(alpha_step)
        self.bank_step_rad  = np.deg2rad(bank_step)
        self.action_space = spaces.Discrete(9)
        
        low  = -np.inf * np.ones(9)
        high =  np.inf * np.ones(9)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 3. 动作映射
        self._action_to_move = {
            0: ( self.alpha_step_rad,  self.bank_step_rad),
            1: ( self.alpha_step_rad,  0.0),
            2: ( self.alpha_step_rad, -self.bank_step_rad),
            3: ( 0.0,                  self.bank_step_rad),
            4: ( 0.0,                  0.0),
            5: ( 0.0,                 -self.bank_step_rad),
            6: (-self.alpha_step_rad,  self.bank_step_rad),
            7: (-self.alpha_step_rad,  0.0),
            8: (-self.alpha_step_rad, -self.bank_step_rad),
        }

        self.phy_state = None 
        self.control_state = None 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置风场时间
        self.wind_manager.reset()

        # 初始化位置 (在高空中心附近)
        x = self.np_random.uniform(0.4, 0.6) * self.domain_size[0]
        y = self.np_random.uniform(0.4, 0.6) * self.domain_size[1]
        z = 0.8 * self.domain_size[2]
        V = 1.0 # 初始速度需要根据无量纲化情况调整
        gamma = 0.0
        chi = self.np_random.uniform(0, 2*np.pi)
        
        self.phy_state = np.array([x, y, z, V, gamma, chi], dtype=np.float32)
        self.control_state = np.array([0.0, 0.0], dtype=np.float32)
        
        return self._get_obs(), {}

    def step(self, action):
        d_alpha, d_bank = self._action_to_move[action]
        
        # 更新控制量
        cur_alpha, cur_phi = self.control_state
        new_alpha = np.clip(cur_alpha + d_alpha, np.deg2rad(-5), np.deg2rad(15))
        new_phi   = np.clip(cur_phi + d_bank, np.deg2rad(-45), np.deg2rad(45))
        self.control_state = np.array([new_alpha, new_phi])

        # 物理积分循环
        dt_rl = 1.0   # 假设时间单位 (如果是无量纲的，此处需调整)
        dt_phy = 0.1
        steps = int(dt_rl / dt_phy)

        for _ in range(steps):
            pos = self.phy_state[0:3]
            # 实时从硬盘读取风速
            wind_vec = self.wind_manager.get_wind(*pos)
            
            self.phy_state = self.physics.integration_step(
                state=self.phy_state,
                control=self.control_state,
                wind_vec=wind_vec,
                dt=dt_phy
            )
            
            # 边界判定 (Z <= 0)
            if self.phy_state[2] <= 0:
                break

        obs = self._get_obs()
        
        # 奖励计算
        reward = -0.1 
        terminated = False
        x, y, z = self.phy_state[0:3]
        
        # 出界判定
        if (z <= 0 or 
            x < 0 or x > self.domain_size[0] or 
            y < 0 or y > self.domain_size[1]):
            terminated = True
            reward = -10.0 # 坠毁或飞出边界
            
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        x, y, z, V, gamma, chi = self.phy_state
        alpha, phi = self.control_state
        theta = gamma + alpha * np.cos(phi)
        
        # 获取当前点的风速作为观测的一部分
        current_wind = self.wind_manager.get_wind(x, y, z)
        # 这里暂用风速分量代替原本的 az, tau
        
        return np.array([x, y, z, V, alpha, phi, theta, current_wind[2], 0.0], dtype=np.float32)

    def close(self):
        self.wind_manager.close()