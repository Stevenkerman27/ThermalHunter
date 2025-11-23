import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GliderEnv(gym.Env):

    def __init__(self, 
                    bank_range: float = 20.0,
                    bank_step: float = 5,
                    alpha_range: float = 12,
                    alpha_step: float = 2.5,
                    az_step = 3,
                    tau_step = 3,
                    verbose: bool = False
                    ):
        super().__init__()
        
        self.bank_angle = bank_range
        self.bank_step = bank_step
        self.AOA_step = alpha_range
        self.alpha_step = alpha_step
        self.verbose = verbose
        self.total_states = az_step * tau_step
        # Action Space
        self.action_space = spaces.Discrete(9) # 3^2 actions over AOA and bankangle 
        # Observation Space
        self.observation_space = spaces.Discrete(self.total_states)

        self._action_to_move = {
            # (0-8), (delta_alpha, delta_bank)
            
            # === 第 0 行: 增加 AOA ===
            0: ( self.alpha_step,  self.bank_step), # 列 0: 增加 Bank
            1: ( self.alpha_step,  0.0),           # 列 1: 不变 Bank
            2: ( self.alpha_step, -self.bank_step), # 列 2: 降低 Bank
            
            # === 第 1 行: 不变 AOA ===
            3: ( 0.0,  self.bank_step), # 列 0: 增加 Bank
            4: ( 0.0,  0.0),           # 列 1: 不变 Bank
            5: ( 0.0, -self.bank_step), # 列 2: 降低 Bank
            
            # === 第 2 行: 降低 AOA ===
            6: (-self.alpha_step,  self.bank_step), # 列 0: 增加 Bank
            7: (-self.alpha_step,  0.0),           # 列 1: 不变 Bank
            8: (-self.alpha_step, -self.bank_step), # 列 2: 降低 Bank
        }

    def step(self, action):
        delta_alpha, delta_bank = self._action_to_move[action]
