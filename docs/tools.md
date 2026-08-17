# 工具

这些工具服务于主滑翔机工作流，但不改变训练和评估的基本契约。

## 奖励权重扫描

```powershell
python reward_sweep.py
python reward_sweep.py --steps 1000 --stats 5 --cpu
```

扫描 `REWARD_W_ACCEL_SWEEP_WEIGHTS` 中的稳态奖励比例。每个权重先串行训练表格 Q，再训练稳态 DQN；全部训练结束后在同一批固定评估场景上运行稳态评估。各模型、日志、传感器统计和评估文件按权重单独保存，汇总 CSV 与图表位于 `trainresult/`。此工具不参与动态实验。

## 单轨迹仿真

`python simulator.py` 加载表格 Q 或稳态 DQN，运行单条轨迹并显示三维路径、传感器、奖励、控制和真空速。策略类型、模型路径、起始帧和最大步数是该脚本内的可视化设置；它以按需读取方式打开风场。

## 动态环境可视化

```powershell
python visualize_dynamic.py
python visualize_dynamic.py --n 1 --max-steps 600
```

该入口串行可视化动态 PPO、动态 DQN、Cruise 和 Random grid。轨迹使用 `memory_mode=False` 的磁盘惰性风场访问，不把风场文件完整加载到内存；策略图和训练曲线不初始化风场。结果写入 `trainresult/dynamic_visualization/`，并按 `ppo`、`dynamic_dqn`、`cruise`、`random_grid` 分目录保存 `trajectory.csv`、`trajectory.png` 和 `summary.csv`；当 `--n` 大于 1 时，轨迹文件按场景编号保存。PPO 和动态 DQN 另外保存适用的 `policy_map.png` 和 `training.png`。

## 分析与可视化

- `analyze_bins.py`：按训练起始帧规则运行随机策略，生成稳态 DQN 所需的 `sensor_stats.json`。
- `plot_dqn_train.py`：从 `trainresult/dqn_train_stats.csv` 生成稳态 DQN 的回报和净爬升图。
- `plot_dqn_slice.py`：在传感器网格上绘制稳态 DQN 的策略切片。
- `readpkl.py`：读取中间表格 Q 文件并绘制离散策略图。
- `plotwind.py`：读取风场的 `buoyancy`，渲染指定 Y 截面的 XZ GIF。
- `test_wind_reading.py`：手工诊断跨 HDF5 文件的时间索引和插值，不是 pytest 测试，且其示例域大小不代表主环境尺度。

`make_pkl.py` 是历史规则策略生成器。它固定使用旧的 7 个滚转分箱，与当前环境状态形状不兼容，不能作为当前训练或评估模型。
