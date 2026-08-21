# 环境

项目提供两套 Gymnasium 环境。二者共用风场读取、极线与风场倍率定义，但飞行模型、控制接口、观测和奖励不同，训练或评估结果不得跨环境直接比较。

## 共同基础

`RBWindField` 按自然序串联 `wind/snapshots_s*.h5`，读取 `tasks/ux`、`tasks/uy`、`tasks/uz` 和 `tasks/buoyancy`。空间风速通过三线性插值得到；`get_wind_at_frame()` 还能在相邻时间帧间线性插值。`memory_mode=True` 会读取整组数据到内存，`False` 保留 HDF5 数据集句柄按需读取。调用方负责在完成后关闭拥有的风场管理器。

`GliderPhysics` 从 `glider.polar` 的 `AoA`、`CL`、`CDtot` 列插值得到气动系数。风速会被统一缩放，使整组三维速度样本的 RMS 与中位迎角、零滚转稳态空速的比值等于 `config.WIND_RMS_TO_TAS_RATIO`。此倍率只改变风速幅值。

起始帧由 `config.sample_start_frame()` 在 `[RESET_START_MIN, RESET_START_MAX)` 中采样，以避开初始瞬态；训练和评估入口均使用该规则。水平坐标周期回绕，高度不回绕。

## 稳态离散环境

`GliderEnv` 使用预计算的稳态空速、下滑角和转弯角速度。动作空间为 9 个离散的 `(迎角增量, 滚转增量)` 组合，每一维为 `-1/0/+1`；发生控制改变的步骤使用 `CONTROL_DRAG_MULTIPLIER` 对应的额外阻力表。

每个 RL 步包含 `N_PHYS_PER_RL` 次积分，风场每 `RL_STEPS_PER_FRAME` 个 RL 步推进一帧。环境返回两种观测形式：

- 表格 Q：`[aoa_idx, bank_idx, w_accel_bin, delta_w_bin]`，后两项经配置分箱和迟滞得到。
- 稳态 DQN：`[aoa_idx, bank_idx, w_accel, delta_w]`，训练包装器将索引缩放并用 `trainresult/sensor_stats.json` 归一化传感器值。

`w_accel` 是一个积分步内垂直风速变化率的平均值，`delta_w` 是两个翼尖垂直风速之差的平均值。奖励为 `u_z + reward_w_accel * w_accel`，没有高度变化或生存项。高度到达域高的 10% 或 90% 时终止，风场帧耗尽时截断。

稳态训练和评估以按需读取模式打开风场；在包装器或调用者的 `reset(options={"resettime": ...})` 中指定起始帧。

## 动态环境

`DynamicGliderEnv` 积分三维位置和地速。每个积分步以 `v_air = v_ground - wind` 计算真实空速，再用极线的升力和阻力以及重力得到加速度。因此空速、下沉率和转弯均由实时积分得到，而非稳态查表值。

动态环境的原生动作是 `[speed, roll]`，均限制在 `[0, 1]`：`speed=0/1` 分别命令最大/最小迎角，`roll=0/1` 分别命令最小/最大滚转角。迎角和滚转先经过一阶响应，再受速率限制。风场帧以 `DYNAMIC_WIND_SECONDS_PER_FRAME` 表示的飞行时间推进，并在帧之间线性插值。

观测为 `[energy_height, total_energy_vario, wingtip_normal_wind_difference, bank]`：总能量高度、其变化率的低通 variometer、翼尖局部风速差沿升力法向的投影，以及当前滚转角。原始四维观测不变；独立命令 `collect_dynamic_observation_stats.py` 用随机批量 rollout 估计各维均值和标准差，并写入 `trainresult/dynamic_observation_normalizer.json`。动态训练优先复用该文件，仅在其缺失时才采样；每个模型工件还会封装所用统计量，推理时必须随模型加载。奖励是本 RL 步的总能量高度增量。位置、地速、迎角、滚转和风场时间以四阶 Runge-Kutta 方法共同积分。高度越过 `DYNAMIC_ALTITUDE_MIN_FRACTION` 或 `DYNAMIC_ALTITUDE_MAX_FRACTION` 时终止；到达最后一个风场帧时截断。低于 `DYNAMIC_MIN_TAS` 时只停用该积分子步的气动力和翼尖滚转传感器计算，但重力继续作用，不结束 episode。若 RK4 候选状态非有限，或 TAS 超过 `DYNAMIC_NUMERICAL_MAX_TAS`，该候选子步被丢弃，episode 以 `numerical_divergence` 终止且该 RL 步奖励为零。`info["termination_reason"]` 记录终止或截断原因。

动态 PPO 直接使用连续动作。动态 DQN 使用 `DynamicDiscreteActionWrapper`，把速度和滚转各离散为 `DYNAMIC_DQN_ACTION_LEVELS` 个等距命令，形成 `DYNAMIC_DQN_ACTION_LEVELS ** 2` 个动作；包装器只转换动作，不改变动力学、观测或奖励。动态训练每次只创建一个内存风场实例；动态评估让所有策略顺序借用同一个只读实例。
