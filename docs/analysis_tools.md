# 分析与可视化工具

## `analyze_bins.py`

以随机策略按训练起始帧规则采样，统计 `w_accel` 与 `delta_w`，并覆盖写入 `sensor_stats.json`。DQN 训练会自动调用它，以使归一化统计与当前风场倍率一致。

## `plot_dqn_train.py`

读取 `trainresult/dqn_train_stats.csv`，绘制回报和净爬升的双轴训练曲线，默认保存为 `trainresult/dqn_train_result.png`。

## `plot_dqn_slice.py`

加载最新 DQN 权重和 `sensor_stats.json`，在传感器均值正负三倍标准差的网格上推理。输出三种攻角索引和三种滚转索引组合的 3x3 决策图到 `trainresult/`。

## `readpkl.py`

批量读取 `q_table/q_table_E_*.pkl`，为中间攻角索引绘制离散策略图，并保存到 `trainresult/policy_<name>.png`。Q 表形状必须符合当前环境的 `(aoa, bank, accel, delta_w, action)` 契约。

## `plotwind.py`

以 `RBWindField(memory_mode=False)` 读取风场，在指定 `Y` 截面上将 `buoyancy` 的 XZ 切片渲染成 GIF。输出文件为项目根目录的 `convective_plumes_xz.gif`。
