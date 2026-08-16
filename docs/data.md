# 数据契约

## 风场

输入位于 `wind/snapshots_s*.h5`，按 `config.natural_key` 的自然序串联。`RBWindField` 读取：

- `tasks/ux`、`tasks/uy`、`tasks/uz`：四维数组 `(T, X, Y, Z)`。
- `tasks/buoyancy`：同样的网格布局，供 `plotwind.py` 绘制切片。
- `tasks/ux.dims[0]['sim_time']`：每帧的物理时间。

环境坐标以 `config.DOMAIN_SIZE` 的物理坐标表示；水平 `x/y` 在环境中周期回绕，高度不回绕。

## 气动极线

`glider.polar` 由 `GliderPhysics` 读取，必须含 `AoA`、`CL`、`CDtot` 列。`config.POLAR_BASE` 不带扩展名，环境据此拼接 `.polar`。

## 传感器统计

`analyze_bins.py` 产生根目录 `sensor_stats.json`。DQN 训练会先自动重建它；DQN 归一化与评估需要其中的 `w_accel.mean/std` 与 `delta_w.mean/std`，缺失时直接报错。

## 模型与结果

- 表格型策略：pickle，形状为 `(AOA_BINS, BANK_BINS, 3, 3, 9)`；最终模型为 `config.SAVE_PATH`。
- DQN 权重：PyTorch `state_dict`，输入维度为 4、动作数为 9；最终模型为 `config.DQN_SAVE_PATH`。
- 动态 PPO 权重：PyTorch `state_dict`，输入为归一化后的 2 个动态传感器，网络侧动作为 2 维 `[-1, 1]`；Gymnasium 包装器将其转换为环境控制的 `[0, 1]^2`。
- 训练结果：`trainresult/`；动态 PPO 的 TensorBoard 日志位于 `trainresult/ppo_runs/`，旧 DQN 的检查点位于 `runs/`。

`test_wind_reading.py` 是手工诊断脚本，不是 pytest 测试。它验证跨 HDF5 文件的时间索引与插值读取；其示例域大小和坐标与主配置不同，不能据此验证主环境的物理尺度。
