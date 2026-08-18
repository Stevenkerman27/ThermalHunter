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
# 可复现实验配置
SEED = 1
EVAL_SEED = 20260816
TRAIN_ALGORITHM = "tabular"

# 训练与评估共用起始帧规则。前 60 帧为风场初始瞬态，不参与采样。
RESET_START_MIN = 60
RESET_START_MAX = 100

def sample_start_frame(rng):
    if hasattr(rng, "integers"):
        return int(rng.integers(RESET_START_MIN, RESET_START_MAX))
    return int(rng.randint(RESET_START_MIN, RESET_START_MAX))

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

# 动态 PPO 环境：连续积分与执行器
DYNAMIC_DT_INTEGRATION = 0.1
DYNAMIC_WIND_SECONDS_PER_FRAME = DT_RL * RL_STEPS_PER_FRAME
DYNAMIC_AOA_MIN_DEG = 0.0
DYNAMIC_AOA_MAX_DEG = 10.0
DYNAMIC_AOA_TIME_CONSTANT = 0.5
DYNAMIC_AOA_RATE_LIMIT_DEG_S = 10.0
DYNAMIC_BANK_TIME_CONSTANT = 1.0
DYNAMIC_BANK_RATE_LIMIT_DEG_S = 20.0
DYNAMIC_MIN_TAS = 5.0
DYNAMIC_NUMERICAL_MAX_TAS = 100.0
DYNAMIC_ALTITUDE_MIN_FRACTION = 0.1
DYNAMIC_ALTITUDE_MAX_FRACTION = 0.9
DYNAMIC_VARIO_TIME_CONSTANT = 1.0
DYNAMIC_ACTION_SATURATION_MARGIN = 0.05
DYNAMIC_CHECKPOINT_INTERVAL = 10000
DYNAMIC_REPORT_EPISODES = 10
DYNAMIC_NORMALIZATION_STEPS = 256
DYNAMIC_BASELINE_SPEED_ACTION = 0.5
DYNAMIC_BASELINE_ROLL_ACTION = 0.5
DYNAMIC_DQN_ACTION_LEVELS = 5
DYNAMIC_NUM_ENVS = 64

# ==========================================
# 风场尺度 (Wind Scale)
# ==========================================
# 实际风场倍率由环境计算，使三维风矢量 RMS / 中位迎角空速等于此值。
WIND_RMS_TO_TAS_RATIO = 0.5
CONTROL_DRAG_MULTIPLIER = 1.1 # 操纵面额外阻力系数 (当 AoA 或 Bank 改变时)
REWARD_W_ACCEL_WEIGHT = 5.0
REWARD_W_ACCEL_SWEEP_WEIGHTS = (1.0, 3.0, 5.0)

# ==========================================
# 训练参数 (Training - Q-Learning)
# ==========================================
ALPHA_START = 0.02        # 初始学习率
ALPHA_END = 0.02          # 最终学习率
GAMMA = 0.98              # 折扣因子
EPSILON_START = 1.0       # 初始探索率
EPSILON_END = 0.01        # 最小探索率
TABULAR_TOTAL_STEPS = 100000
Q_TABLE_NAME = "q_table_v0.pkl"
SAVE_PATH = os.path.join(Q_TABLE_DIR, Q_TABLE_NAME)

# ==========================================
# 训练参数 (Training - DQN)
# ==========================================
DQN_LR = 6e-5
DQN_GAMMA = 0.995
DQN_BATCH_SIZE = 128
DQN_BUFFER_SIZE = 100000
DQN_TARGET_UPDATE_INTERVAL = 10 # 用于旧脚本，新脚本使用 DQN_TARGET_FREQ
DQN_HIDDEN_SIZE = 32
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.05
DQN_TOTAL_TIMESTEPS = 50000
DQN_TAU = 1.0
DQN_TARGET_FREQ = 4000
DQN_EXPLORATION_FRACTION = 0.5
DQN_LEARNING_STARTS = 2000
DQN_TRAIN_FREQ = 4
DQN_TORCH_THREADS = 1
DQN_ACTION_MARGIN_K = 0.1  # 动态阈值系数
DQN_ACTION_MARGIN_MIN = 0.02 # 固定最小阈值
DQN_SAVE_PATH = os.path.join(Q_TABLE_DIR, "dqn_model.pth") # 用于 train_dqn.py 保存路径
SENSOR_STATS_EPISODES = 20

# ==========================================
# 训练参数 (Training - Dynamic PPO)
# ==========================================
PPO_TOTAL_TIMESTEPS = 100000
PPO_LEARNING_RATE = 3e-4
# Per-environment rollout length. With DYNAMIC_NUM_ENVS=64, each PPO update
# retains the original 1,024 total transitions.
PPO_NUM_STEPS = 16
PPO_NUM_MINIBATCHES = 32
PPO_UPDATE_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_COEF = 0.2
PPO_ENT_COEF = 0.01
PPO_ENT_COEF_FINAL = 0.0
PPO_CLIP_VLOSS = False
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_TORCH_THREADS = 1
PPO_SQUASH_EPSILON = 1e-6

# ==========================================
# 训练参数 (Training - Dynamic DQN)
# ==========================================
DYNAMIC_DQN_TOTAL_TIMESTEPS = 80000
DYNAMIC_DQN_LEARNING_RATE = 5e-5
DYNAMIC_DQN_GAMMA = 0.99
DYNAMIC_DQN_BATCH_SIZE = 64
DYNAMIC_DQN_BUFFER_SIZE = 100000
DYNAMIC_DQN_EPSILON_START = 1.0
DYNAMIC_DQN_EPSILON_END = 0.2
DYNAMIC_DQN_EXPLORATION_FRACTION = 0.5
DYNAMIC_DQN_LEARNING_STARTS = 5000
DYNAMIC_DQN_TRAIN_FREQ = 10
DYNAMIC_DQN_TARGET_FREQ = 5000
DYNAMIC_DQN_TAU = 0.4
DYNAMIC_DQN_TORCH_THREADS = 1

# 追踪特定的 Q 值索引 (s_aoa, s_bank, s_accel, s_delta, action)
TRACK_INDICES = [(2, 3, 1, 1, 4), (2, 3, 2, 2, 0), (2, 3, 0, 0, 8)]

# ==========================================
# 评估参数 (Evaluation)
# ==========================================
N_EVAL_EPISODES = 80
