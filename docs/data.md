# 数据与产物

## 输入

风场文件位于 `wind/snapshots_s*.h5`，按自然序拼接。当前数据来自 Rayleigh--Benard 对流在 Rayleigh number `Ra = 5e7` 下的直接数值模拟（DNS）；它是受控的数值风场数据，不是实飞测量。每个文件必须含有：

- `tasks/ux`、`tasks/uy`、`tasks/uz`：形状为 `(T, X, Y, Z)` 的速度场。
- `tasks/buoyancy`：同一网格布局，供 `plotwind.py` 使用。
- `tasks/ux.dims[0]['sim_time']`：帧时间坐标。

环境把 `config.DOMAIN_SIZE` 映射到网格坐标；`x/y` 周期回绕，高度由环境边界规则处理。`glider.polar` 必须有 `AoA`、`CL` 和 `CDtot` 列，且 `config.POLAR_BASE` 不带扩展名。

## 传感器统计

`trainresult/sensor_stats.json` 含 `w_accel.mean/std` 与 `delta_w.mean/std`。稳态 DQN 训练会先重建该文件，随后训练和评估都用它归一化连续传感器；奖励扫描的对应统计也写入 `trainresult/`。统计不存在或标准差为零时，生成或加载环节会报错。

## 模型和结果

- 稳态表格 Q：pickle，形状由当前稳态离散观测与 9 动作空间决定，默认最终模型为 `q_table/q_table_dqn_bins_v1.pkl`。该表必须与 `config.BINS_W_ACCEL` 和 `config.BINS_DELTA_W` 一起使用。
- 稳态 DQN：PyTorch `state_dict`，输入 4 维归一化观测、输出 9 个 Q 值，默认最终模型为 `q_table/dqn_model_tuned_v1.pth`。
- 动态 PPO：PyTorch 工件，输入 4 维归一化动态观测，网络动作是 2 维 `[-1, 1]`，包装器再映射到环境的 `[0, 1]^2`。
- 动态 DQN：PyTorch 工件，输入 4 维归一化动态观测、输出 `DYNAMIC_DQN_ACTION_LEVELS ** 2` 个 Q 值。

训练 CSV、评估 CSV、图表和 TensorBoard 日志均写入项目内的 `trainresult/`、`runs/` 或 `q_table/`；各模块的默认文件名见[训练](training.md)和[评估](evaluation.md)。现有模型和产物不要求与后续代码兼容。
