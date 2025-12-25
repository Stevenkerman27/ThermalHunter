import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class GridSwimmerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, grid_size=10, render_mode=None, reward_reach = 10, reward_step = 2):
        super().__init__()
        
        self.grid_size = grid_size
        self.window_size = 500  # 渲染窗口大小
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.reward_reach = reward_reach
        self.reward_step = reward_step

        # --- 动作空间 ---
        # 0: 上, 1: 右, 2: 下, 3: 左
        self.action_space = spaces.Discrete(4)

        self.obs_range = grid_size
        self.observation_space = spaces.MultiDiscrete([self.obs_range, self.obs_range])

        # 动作对应的坐标变化 (dx, dy)
        # 假设 (0,0) 在左下角 -> Up 是 y+1
        self._action_to_direction = {
            0: np.array([0, 1]),  # Up
            1: np.array([1, 0]),  # Right
            2: np.array([0, -1]), # Down
            3: np.array([-1, 0]), # Left
        }


    # 辅助函数：计算环形世界下的最短曼哈顿距离
    # 用于计算 Reward，判断是否真正靠近目标
    def _get_toroidal_delta(self, pos1, pos2):
        # 计算原始差值
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        # X 轴处理：如果距离超过一半，就往反方向“折叠”
        if dx > self.grid_size / 2:
            dx -= self.grid_size
        elif dx < -self.grid_size / 2:
            dx += self.grid_size
            
        # Y 轴处理
        if dy > self.grid_size / 2:
            dy -= self.grid_size
        elif dy < -self.grid_size / 2:
            dy += self.grid_size
            
        return np.array([dx, dy])
    
    def _calculate_toroidal_dist(self, pos1, pos2):
        delta = self._get_toroidal_delta(pos1, pos2)
        # 曼哈顿距离 = |dx| + |dy|
        return np.abs(delta).sum()
    
    def _get_obs(self):
        # 获取最短向量 (例如 -1, 0)
        delta = self._get_toroidal_delta(self._agent_pos, self._target_pos)
        
        # 加上 Offset 变成 Q-Table 索引 (例如 11, 12)
        offset = self.grid_size // 2
        return (delta + offset).astype(int)

    def _get_info(self):
        # 计算曼哈顿距离 (Manhattan Distance) 作为理论最短距离
        dist = self._calculate_toroidal_dist(self._agent_pos, self._target_pos)
        opt_reward = self.reward_reach + dist * self.reward_step - dist
        return {"distance": dist, "max reward": opt_reward}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 随机生成 Agent 和 Target 位置
        self._agent_pos = self.np_random.integers(0, self.grid_size, size=2)
        self._target_pos = self.np_random.integers(0, self.grid_size, size=2)
        
        # 确保它们不重叠
        while np.array_equal(self._agent_pos, self._target_pos):
            self._target_pos = self.np_random.integers(0, self.grid_size, size=2)

        self.step_count = 0
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.step_count += 1
        
        # 记录旧距离 (用于计算 Reward)
        dist_old = self._calculate_toroidal_dist(self._agent_pos, self._target_pos)
        
        direction = self._action_to_direction[action]
        new_pos = self._agent_pos + direction
        
        # --- 核心修改：取模运算实现无限边界 ---
        # 25 -> 0, -1 -> 24
        self._agent_pos = new_pos % self.grid_size
        
        # 计算新距离
        dist_new = self._calculate_toroidal_dist(self._agent_pos, self._target_pos)
        
        terminated = False
        truncated = False
        reward = -1  # 基础步数惩罚
        
        # 奖励计算：靠近给正分，远离给负分 (无需再处理撞墙逻辑)
        reward += (dist_old - dist_new) * self.reward_step
        
        # 到达判断
        if np.array_equal(self._agent_pos, self._target_pos):
            terminated = True
            reward += self.reward_reach
        
        # 超时判断
        if self.step_count >= self.grid_size * 10:
            truncated = True
            
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        pix_square_size = self.window_size / self.grid_size

        # 绘制 Target (红色方块)
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                self._target_pos[0] * pix_square_size,
                (self.grid_size - 1 - self._target_pos[1]) * pix_square_size, # Y轴翻转适应屏幕坐标
                pix_square_size,
                pix_square_size,
            ),
        )
        
        # 绘制 Agent (蓝色方块)
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                self._agent_pos[0] * pix_square_size,
                (self.grid_size - 1 - self._agent_pos[1]) * pix_square_size, # Y轴翻转
                pix_square_size,
                pix_square_size,
            ),
        )

        # 画网格线
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas, 0, (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=2
            )
            pygame.draw.line(
                canvas, 0, (pix_square_size * x, 0), (pix_square_size * x, self.window_size), width=2
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()