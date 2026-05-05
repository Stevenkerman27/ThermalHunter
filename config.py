import os
import numpy as np
import re

# ==========================================
# 辅助函数 (Utils)
# ==========================================
def natural_key(string_):
    """用于文件名等含有数字的字符串自然排序"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# ==========================================
# 路径配置 (Paths)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WIND_DIR = os.path.join(BASE_DIR, 'wind')
Q_TABLE_DIR = os.path.join(BASE_DIR, 'q_table')
TRAIN_RESULT_DIR = os.path.join(BASE_DIR, 'trainresult')

glider_data = ["LS-8 (15m)", 346, 185, 80, -0.59, 115, -0.76, 173, -2.00, 10.5, 190, 108, 240]
#滑翔机名称，参考重量，压舱水重量，速度，下沉率，速度，下沉率，速度，下沉率，翼面积.....

# 确保目录存在
for d in [Q_TABLE_DIR, TRAIN_RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# 物理与飞行器参数 (Physics & Aircraft)
# ==========================================
POLAR_BASE = "glider"
MASS = 2.0         # 质量 (kg)
AREA = 0.3         # 迎风面积 (m^2)
WINGSPAN = 10.0    # 翼展 (m)
DOMAIN_SIZE = (1000.0, 1000.0, 1000.0)

# ==========================================
# 环境与状态空间 (Environment & State)
# ==========================================
# 训练重置时间范围 (风场帧索引)
RESET_TIME_MIN = 30
RESET_TIME_MAX = 300

# 坡度控制 (Bank)
BANK_MIN_DEG = -20.0
BANK_MAX_DEG = 20.0
BANK_STEP_DEG = 10.0
BANK_BINS = int((BANK_MAX_DEG - BANK_MIN_DEG) / BANK_STEP_DEG) + 1 

# 攻角控制 (AoA)
AOA_MIN_DEG = 0.0
AOA_MAX_DEG = 9.0
AOA_STEP_DEG = 3.0
AOA_BINS = int((AOA_MAX_DEG - AOA_MIN_DEG) / AOA_STEP_DEG) + 1 

# 传感器分箱 (Sensor Bins)
BINS_W_ACCEL = np.array([-0.3, 0.3])
BINS_DELTA_W = np.array([-0.23, 0.23])
HYSTERESIS_PCT = 0.1  # 施密特触发器迟滞比例

# 时间控制
DT_RL = 1.0               # RL 控制步长 (s)
N_PHYS_PER_RL = 2         # 每个 RL 步内的物理积分次数
RL_STEPS_PER_FRAME = 2    # 多少步 RL 更新一次风场数据帧

# ==========================================
# 奖励函数参数 (Reward)
# ==========================================
WIND_AMPF = 12.0          # 风场放大系数
REWARD_LAMBDA = 0.2       # 高度变化奖励权重
REWARD_SURVIVE = 0.0      # 每步生存奖励
CONTROL_DRAG_MULTIPLIER = 1.1 # 操纵面额外阻力系数 (当 AoA 或 Bank 改变时)

# ==========================================
# 训练参数 (Training - Q-Learning)
# ==========================================
ALPHA = 0.04              # 学习率
GAMMA = 0.999             # 折扣因子
EPSILON_START = 1.0       # 初始探索率
EPSILON_END = 0.01        # 最小探索率
EPISODES = 8000           # 总训练集数
SAVE_INTERVAL = 2000      # 模型保存间隔
Q_TABLE_NAME = "q_table_v0.pkl"
SAVE_PATH = os.path.join(Q_TABLE_DIR, Q_TABLE_NAME)

# ==========================================
# 训练参数 (Training - DQN)
# ==========================================
DQN_LR = 1e-4
DQN_GAMMA = 0.99
DQN_BATCH_SIZE = 32
DQN_BUFFER_SIZE = 100000
DQN_TARGET_UPDATE_INTERVAL = 10
DQN_HIDDEN_SIZE = 32
DQN_SAVE_PATH = os.path.join(Q_TABLE_DIR, "dqn_model.pth")
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.05
DQN_EPISODES = 1500

# 追踪特定的 Q 值索引 (s_aoa, s_bank, s_accel, s_delta, action)
TRACK_INDICES = [(2, 3, 1, 1, 4), (2, 3, 2, 2, 0), (2, 3, 0, 0, 8)]

# ==========================================
# 评估参数 (Evaluation)
# ==========================================
N_EVAL_EPISODES = 300     # 评估时的集数