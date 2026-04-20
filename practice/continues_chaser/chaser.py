import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RelativePursuitEnv(gym.Env):
    """
    一个Agent追踪移动目标的2D连续环境。
    状态不包含绝对坐标，仅包含相对观测值。
    """
    def __init__(self, max_steps=400):
        super(RelativePursuitEnv, self).__init__()

        # 参数设置
        self.v_a = 2       # Agent速度
        self.v_t = 1.6        # 目标速度 (v_t < v_a)
        self.dt = 0.1         # 步长
        self.d_min = 0.5      # 捕获距离
        self.d_max = 50.0     # 丢失距离
        self.reward_reach = 20
        self.reward_fail = -20
        self.reward_step = -(self.v_a-self.v_t)
        self.max_steps = max_steps

        # 动作空间：[-pi, pi]，代表下一时刻移动方向与 Line of Sight 的夹角
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)

        # 观测空间：[与LOS夹角, 目标相对角速度]（经过tanh变换后有界）
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )

        self.state = None
        self.steps_beyond_done = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 随机初始化物理位置（内部使用，不暴露给agent）
        r = self.np_random.uniform(10, 20)
        angle = self.np_random.uniform(0, 2 * np.pi)
        
        self.agent_pos = np.array([0.0, 0.0])
        self.target_pos = np.array([r * np.cos(angle), r * np.sin(angle)])
        
        # 目标匀速直线运动的方向
        side = self.np_random.choice([-1, 1])
        self.target_heading = self._normalize_angle(angle + np.pi + side * self.np_random.uniform(np.pi/3, 2*np.pi/3))
        # Agent初始朝向指向目标
        self.agent_heading = angle
        
        self.current_step = 0
        self.prev_dist = r
        self.prev_los_angle = angle
        
        return self._get_obs(), {}

    def _get_obs(self):
        # 计算相对距离矢量
        rel_pos = self.target_pos - self.agent_pos
        dist = np.linalg.norm(rel_pos)
        
        # 当前视线角 (Line of Sight angle)
        los_angle = np.arctan2(rel_pos[1], rel_pos[0])
        
        # 上一步移动方向与LOS的夹角 (归一化到 -pi, pi)
        phi = self._normalize_angle(los_angle - self.agent_heading)
        
        # 目标移动产生的角速度 (LOS的变化率)
        # 注意：在reset后的第一步，delta_los可能为0
        delta_los = self._normalize_angle(los_angle - self.prev_los_angle)
        omega = delta_los / self.dt
        
        self.prev_los_angle = los_angle
        self.prev_dist = dist
        
        # phi 的范围是 [-π, π]，除以 π 归一化到 [-1, 1]
        # omega 需要缩放，假设最大角速度为 5 rad/s，然后 tanh 变换
        max_omega = 5.0  # 最大角速度假设为 5 rad/s
        phi = phi / np.pi # phi 在 [-π, π] 范围内，除以 π 归一化
        omega_tanh = np.tanh(omega / max_omega)  # omega 缩放后 tanh
        
        return np.array([phi, omega_tanh], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        # 获取动作：agent速度方向与当前LOS的夹角
        # action 是 [-pi, pi] 之间的连续值
        rel_action_angle = action[0]
        
        # 计算当前的 LOS 角
        rel_pos = self.target_pos - self.agent_pos
        los_angle = np.arctan2(rel_pos[1], rel_pos[0])
        
        # 更新 Agent 朝向
        self.agent_heading = self._normalize_angle(los_angle + rel_action_angle)
        
        # 物理运动更新
        # Agent 移动
        self.agent_pos[0] += self.v_a * np.cos(self.agent_heading) * self.dt
        self.agent_pos[1] += self.v_a * np.sin(self.agent_heading) * self.dt
        
        # 目标移动 (匀速直线)
        self.target_pos[0] += self.v_t * np.cos(self.target_heading) * self.dt
        self.target_pos[1] += self.v_t * np.sin(self.target_heading) * self.dt
        
        # 计算新距离和奖励
        new_dist = np.linalg.norm(self.target_pos - self.agent_pos)
        
        # 接近率奖励 (正值表示正在靠近)
        approach_rate = (self.prev_dist - new_dist) / self.dt
        reward = approach_rate + self.reward_step
        
        # 终止条件检查
        terminated = False
        truncated = False
        
        if new_dist < self.d_min:
            reward += self.reward_reach  # 捕获奖励
            terminated = True
            
        if self.current_step >= self.max_steps:
            reward += self.reward_fail
            truncated = True

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, {}

    def _normalize_angle(self, angle):
        """将角度映射到 [-pi, pi]"""
        while angle > np.pi: angle -= 2.0 * np.pi
        while angle < -np.pi: angle += 2.0 * np.pi
        return angle