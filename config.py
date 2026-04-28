import os
import numpy as np

# ==========================================
# 路径配置 (Paths)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WIND_DIR = os.path.join(BASE_DIR, 'wind')
Q_TABLE_DIR = os.path.join(BASE_DIR, 'q_table')
TRAIN_RESULT_DIR = os.path.join(BASE_DIR, 'trainresult')

# 确保目录存在
for d in [Q_TABLE_DIR, TRAIN_RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# 物理与飞机参数 (Physics & Aircraft)
# ==========================================
POLAR_BASE = "glider"
MASS = 2.0         # 质量 (kg)
AREA = 0.3         # 翼面积 (m^2)
WINGSPAN = 10.0    # 翼展 (m)
DOMAIN_SIZE = (1000.0, 1000.0, 1000.0)

# ==========================================
# 环境与状态空间 (Environment & State)
# ==========================================
# 坡度控制 (Bank)
BANK_MIN_DEG = -15.0
BANK_MAX_DEG = 15.0
BANK_STEP_DEG = 5.0
BANK_BINS = int((BANK_MAX_DEG - BANK_MIN_DEG) / BANK_STEP_DEG) + 1  # 7 bins

# 攻角控制 (AoA)
AOA_FIXED_DEG = 9.0

# 传感器分箱 (Sensor Bins)
BINS_W_ACCEL = np.array([-0.2, 0.2])
BINS_DELTA_W = np.array([-0.2, 0.2])
HYSTERESIS_PCT = 0.1  # 施密特触发器迟滞比例

# 时间控制
DT_RL = 1.0               # RL 控制步长 (s)
N_PHYS_PER_RL = 2         # 每个 RL 步内的物理积分次数
RL_STEPS_PER_FRAME = 2    # 多少步 RL 更新一次风场数据帧

# ==========================================
# 奖励函数参数 (Reward)
# ==========================================
WIND_AMPF = 12.0          # 风场放大系数
REWARD_LAMBDA = 2.0       # 高度变化奖励权重
REWARD_SURVIVE = 0.0      # 每步生存奖励

# ==========================================
# 训练参数 (Training - Q-Learning)
# ==========================================
ALPHA = 0.04              # 学习率
GAMMA = 0.999             # 折扣因子
EPSILON_START = 1.0       # 初始探索率
EPSILON_END = 0.01        # 最小探索率
EPISODES = 6000           # 总训练集数
SAVE_INTERVAL = 2000      # 模型保存间隔
Q_TABLE_NAME = "q_table_v0.pkl"
SAVE_PATH = os.path.join(Q_TABLE_DIR, Q_TABLE_NAME)

# 追踪特定的 Q 值索引 (s_bank, s_accel, s_delta, action)
TRACK_INDICES = [(3, 1, 1, 1), (3, 2, 2, 0), (3, 0, 0, 2)]

# ==========================================
# 评估参数 (Evaluation)
# ==========================================
N_EVAL_EPISODES = 200     # 评估时的集数