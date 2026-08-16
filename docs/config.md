# `config.py`

## 职责

`config.py` 是滑翔机项目的集中配置源，定义路径、飞行器与环境参数、奖励、表格型训练、DQN 训练和评估参数。主链路中的模块应从此处读取共享可调参数。

## 分组

- 路径：`BASE_DIR`、`WIND_DIR`、`Q_TABLE_DIR`、`TRAIN_RESULT_DIR`。
- 飞行器与物理域：`POLAR_BASE`、`MASS`、`AREA`、`WINGSPAN`、`DOMAIN_SIZE`。
- 观测与控制：攻角/滚转离散范围、传感器分箱、迟滞比例、控制步长与风场换帧频率。
- 实验：训练算法、训练/评估种子、训练与评估共用的随机起始帧范围及评估 episode 数。
- 奖励与尺度：`REWARD_W_ACCEL_WEIGHT` 是垂直风加速度的单次训练权重，垂直风速度系数固定为 1；`REWARD_W_ACCEL_SWEEP_WEIGHTS` 定义批量比较的权重。尺度配置包括目标三维风场 RMS/中位迎角空速比与操纵阻力系数。
- 训练：两类算法的总步数上限、DQN 超参数、模型路径、传感器统计采样量和评估 episode 数。

导入该模块会创建 `q_table/` 与 `trainresult/`。风场实际倍率是环境从数据计算出的派生量，不应在调用方手写。除工具自身的历史常量外，主工作流不应复制这些配置。
