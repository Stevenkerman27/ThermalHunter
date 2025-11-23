import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ElevatorEnv(gym.Env):

    def __init__(self, 
                    n_floors: int = 5,
                    passengers: int = 5,
                    reward_deliver: int = 20,
                    reward_pickup: int = 10,
                    reward_time_step: int = -1,
                    reward_falseopen: int = -5,
                    reward_outrange: int = -5,
                    verbose: bool = False,
                    ):
        super().__init__()  # 继承时需要调用
        
        self.n_floors = n_floors
        self.reward_deliver = reward_deliver
        self.reward_pickup = reward_pickup
        self.reward_time_step = reward_time_step
        self.reward_falseopen = reward_falseopen
        self.reward_outrange = reward_outrange
        self.passengers = passengers
        self.verbose = verbose
        # 定义动作空间 (Action Space) 
        self.action_space = spaces.Discrete(3)

        self._action_to_move = {
            0: -1,  # Down
            1: 0,   # Stop / Stay
            2: 1    # Up
        }

        self._action_to_name = {
            0: "Move Down",
            1: "Stop/Open",
            2: "Move Up"
        }

        # 定义观测空间 (Observation Space)
        self.n_floor_states = n_floors                   # 5
        self.n_car_call_states = 2**n_floors             # 2^5 = 32
        self.n_hall_up_states = 2**(n_floors - 1)      # 2^4 = 16
        self.n_hall_down_states = 2**(n_floors - 1)    # 2^4 = 16

        # 总状态数
        self.total_states = (
            self.n_floor_states
            * self.n_car_call_states
            * self.n_hall_up_states
            * self.n_hall_down_states
        )
        print("Total stats: " + str(self.total_states))
        self.observation_space = spaces.Discrete(self.total_states)
        # 当前楼层, 0-indexed
        self._current_floor = 0 

        #等待乘客
        self._passengers_waiting = np.zeros((n_floors, n_floors), dtype=np.int32)
        #电梯内乘客
        self._passengers_in_car = np.zeros(n_floors, dtype=np.int32)

    def _get_obs(self) -> int:
        car_calls_array = (self._passengers_in_car > 0).astype(np.int8) #内部按钮
        # 推导上行按钮
        hall_up_calls_array = np.zeros(self.n_floors - 1, dtype=np.int8)
        for i in range(self.n_floors - 1):
            if self._passengers_waiting[i, (i+1):].sum() > 0:  #第i层，有想去i层之上的就按上
                hall_up_calls_array[i] = 1

        # 推导下行按钮
        hall_down_calls_array = np.zeros(self.n_floors - 1, dtype=np.int8)
        for i in range(1, self.n_floors):
            if self._passengers_waiting[i, :i].sum() > 0: #第i层，有想去i层之下的就按下
                hall_down_calls_array[i - 1] = 1

        #将3个数组分别转为整数
        car_calls_int = self._binary_array_to_int(car_calls_array)
        up_calls_int = self._binary_array_to_int(hall_up_calls_array)
        down_calls_int = self._binary_array_to_int(hall_down_calls_array)

        # 组合成唯一的 Q-table 索引
        obs = self._current_floor

        obs = obs * self.n_car_call_states + car_calls_int
        obs = obs * self.n_hall_up_states + up_calls_int
        obs = obs * self.n_hall_down_states + down_calls_int
        return obs

    def _binary_array_to_int(self, bin_array):
        """辅助函数: 将 [1, 0, 1] 转换为 5"""
        res = 0
        for bit in bin_array:
            res = (res << 1) | bit
        return res

    def _generate_random_request(self, passengers):
        for i in range(passengers):
            start_floor = self.np_random.integers(0, self.n_floors)
            
            end_floor = self.np_random.integers(0, self.n_floors)
            
            while end_floor == start_floor:
                end_floor = self.np_random.integers(0, self.n_floors)

            self._passengers_waiting[start_floor, end_floor] += 1
            if self.verbose:
                print(f"  -> Passenger {i+1}: from {start_floor} to {end_floor}")

    def reset(self, seed=None, options=None):
        # 调用父类的 reset 来处理随机种子
        super().reset(seed=seed)

        self._last_action = -1
        
        # 将电梯状态重置为“空闲”, 在一个随机楼层开始
        self._current_floor = self.np_random.integers(0, self.n_floors)
    
        # 清空“真实”的物理引擎
        self._passengers_waiting.fill(0)
        self._passengers_in_car.fill(0)

        self._generate_random_request(self.passengers)
            
        observation = self._get_obs()
    
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        self._last_action = action 
        move = self._action_to_move[action]  # 'move' 将是 -1, 0, 或 1
        reward = self.reward_time_step 
        if move == 0:
            # --- 乘客下车 ---
            passengers_getting_off = self._passengers_in_car[self._current_floor]
            if passengers_getting_off > 0:
                self._passengers_in_car[self._current_floor] = 0
                reward += passengers_getting_off * self.reward_deliver

            # --- 上行乘客 ---
            passengers_on_up_count = self._passengers_waiting[self._current_floor, self._current_floor+1:].sum()
            if passengers_on_up_count > 0:
                for j in range(self._current_floor+1, self.n_floors):
                    num = self._passengers_waiting[self._current_floor, j]
                    if num > 0:
                        self._passengers_in_car[j] += num
                        self._passengers_waiting[self._current_floor, j] = 0
                reward += passengers_on_up_count * self.reward_pickup

            # --- 下行乘客 ---
            passengers_on_down_count = self._passengers_waiting[self._current_floor, :self._current_floor].sum()
            if passengers_on_down_count > 0:
                for j in range(0, self._current_floor):
                    num = self._passengers_waiting[self._current_floor, j]
                    if num > 0:
                        self._passengers_in_car[j] += num
                        self._passengers_waiting[self._current_floor, j] = 0
                reward += passengers_on_down_count * self.reward_pickup

            # --- false-open 判断（完全没有乘客上下）---
            if (passengers_getting_off == 0 
                and passengers_on_up_count == 0 
                and passengers_on_down_count == 0):
                reward += self.reward_falseopen

        else:
            # ===== 非法动作：顶层上行 or 底层下行 =====
            if (self._current_floor == 0 and move == -1) or \
            (self._current_floor == self.n_floors - 1 and move == 1):
                reward += self.reward_outrange
            else:
                self._current_floor += move

        # 检查任务是否结束 (Terminated),如果电梯里没人了，并且外面也没人等了
        total_requests = self._passengers_waiting.sum() + self._passengers_in_car.sum()
        terminated = (total_requests == 0)
        truncated = False 
        
        # 获取观测值和信息
        info = self._get_info() 
        observation = self._get_obs() # _get_obs() 会翻译成按钮
        
        return observation, reward, terminated, truncated, info

    def _get_info(self):
        # 查找动作名字 
        action_name = "N/A (Reset)" # 默认值 (如果 _last_action 还是 -1)
        if self._last_action in self._action_to_name:
            action_name = self._action_to_name[self._last_action]
        return {
            "current_floor": self._current_floor,
            "passengers_in_car": self._passengers_in_car.copy(),
            "passengers_waiting": self._passengers_waiting.copy(),
            "action_name": action_name
        }
    
if __name__ == "__main__":
    print("--- 开始测试电梯环境 ---")

    try:
        env = ElevatorEnv(n_floors=5, verbose=True)
        print(f"环境创建成功!")
        print(f"动作空间: {env.action_space}")
        print(f"观测空间总数: {env.observation_space.n}")
    except Exception as e:
        print(f"环境创建失败: {e}")
        exit() # 如果创建失败，后续无法进行

    # 2. 测试 reset() 函数
    print("\n--- 测试 reset() 函数 (调用5次) ---")
    
    for i in range(5):
        print(f"\n[Reset 调用 {i+1}]")
        
        # 调用 reset，并传入一个 seed 以便复现 (好习惯)
        obs, info = env.reset() 
        
        print(f"  -> 返回的 Observation (Q-table 索引): {obs}")
        print(f"  -> 返回的 Info (人类可读状态):")
        
        # 格式化打印 info 字典
        for key, value in info.items():
            print(f"     {key}: {value}")
            
        # 检查项
        if not isinstance(obs, (int, np.integer)):
             print(f"失败: Observation 不是一个整数!")
        
        # 检查是否至少有一个请求
        request_sum = (
            info['passengers_in_car'].sum() + 
            info['passengers_waiting'].sum()
        )
        if request_sum == 0:
            print(f"失败: Reset 后没有任何请求!")
        
    print("\n\n--- 基础测试完成 ---")
    print("\n--- 测试step() 函数 ---")
    action = env.action_space.sample() # 选一个随机动作
    print(f"执行一个随机动作: {action}")
    env.step(action) 
    info = env._get_info()
    for key, value in info.items():
        print(f"     {key}: {value}")