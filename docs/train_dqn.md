# `train_dqn.py`

## 职责

用 CleanRL 风格的 DQN 在连续传感器观测上训练滑翔机策略。命令入口为 `python train_dqn.py`；`tyro` 从 `Args` 提供可选命令行参数，默认值来自 `config.py`。

## 组成

- 注册 `GliderContinuous-v0`，其底层为 `GliderEnv(continuous_obs=True)`。
- 训练前串行运行 `collect_sensor_stats()`，以当前风场倍率和训练帧规则重建 `sensor_stats.json`。
- `GliderWrapper` 在 reset 时按统一规则随机设置训练起始帧；采样范围避开风场初始瞬态。
- `reward_w_accel` 控制垂直风加速度在环境奖励中的系数；基准垂直风速度系数为 1。
- `normalize_state` 将前两个离散索引缩放，并使用传感器统计标准化两个传感器量；统计缺失直接报错。
- `QNetwork`：4 维输入、两层 ReLU 隐藏层、9 个动作 Q 值输出。

## 训练与产物

训练使用单环境 `SyncVectorEnv`、内置经验回放、epsilon-greedy、MSE TD 损失和目标网络软/硬更新。`num_envs` 必须为 1；统计环境关闭后才创建训练环境。

每个完成的 episode 将步数、回报、净爬升写入 `trainresult/dqn_train_stats.csv`；TensorBoard 日志和阶段检查点写入 `runs/<run_name>/`。启用模型保存时，最终权重额外写入 `config.DQN_SAVE_PATH`。训练曲线仅保存图片，不打开交互窗口。
