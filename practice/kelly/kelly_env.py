import gymnasium as gym
from gymnasium import spaces
import numpy as np

class KellyBettingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 win_prob=0.6, 
                 start_balance=1000, 
                 min_balance=1.0, 
                 target_balance=10000.0, 
                 max_steps=40,
                 step_penalty=0.01):
        
        super().__init__()
        
        self.win_prob = win_prob
        self.start_balance = start_balance
        self.min_balance = min_balance
        self.target_balance = target_balance
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        
        # 动作空间：下注比例 [0, 1]
        self.action_space = spaces.Box(low=0.0, high=0.999, shape=(1,), dtype=np.float32)
        
        # 观测空间：当前本金
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        
        # 初始化内部状态
        self.balance = self.start_balance
        self.current_step = 0
        self.wins = 0
        self.losses = 0
        self.last_fraction = 0.0
        self.last_win = False

    def _get_obs(self):
        """返回当前的观测值 (本金)"""
        return np.array([np.log(max(self.balance, 1e-8))], dtype=np.float32)

    def _get_info(self):
        """返回当前的统计信息 (胜负次数)"""
        return {
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.wins / max(1, self.wins + self.losses), # 避免除以0
            "fraction": self.last_fraction,
            "win": self.last_win
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置状态
        self.balance = self.start_balance
        self.current_step = 0
        self.wins = 0
        self.losses = 0
        
        self.last_fraction = 0.0
        self.last_win = False
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        
        # 1. 动作处理
        high_bound = float(self.action_space.high[0])
        fraction = float(np.clip(action[0], 0.0, high_bound))
        self.last_fraction = fraction
        bet_amount = self.balance * fraction
        
        old_balance = self.balance
        
        # 2. 胜负逻辑
        win = np.random.random() < self.win_prob
        self.last_win = win
        
        if win:
            self.balance += bet_amount # 赢：+1倍赌注 (总共拿回2倍)
            self.wins += 1
        else:
            self.balance -= bet_amount # 输：-1倍赌注
            self.losses += 1
            
        # 3. 终止与截断判定
        terminated = False
        truncated = False
        
        if self.balance <= self.min_balance:
            terminated = True
            self.balance = self.min_balance # 钳位
        elif self.balance >= self.target_balance:
            terminated = True
            
        if self.current_step >= self.max_steps:
            truncated = True

        # 4. Reward 计算 (对数效用增量 - 时间惩罚)
        safe_old = max(old_balance, 1e-8)
        safe_new = max(self.balance, 1e-8)
        reward = (np.log(safe_new) - np.log(safe_old)) - self.step_penalty
        
        # 5. 返回标准五元组，使用 helper 生成 obs 和 info
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        print(f"Step: {self.current_step:4d} | Balance: {self.balance:10.4f} | Wins: {self.wins} | Losses: {self.losses}")