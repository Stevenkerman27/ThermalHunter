# 项目架构
simulator.py为可视化滑翔机轨迹的模拟器
eval.py批量评估随机和训练后策略的性能 (已模块化，支持在训练脚本中直接调用)
plot_wing.py绘制风场切片的gif图
readpkl.py读取训练的策略并绘图
make_pkl.py制作理想策略
test_wind_reading.py为测试风场数据的代码
glider_discrete_simp.py为环境定义主文件 (已通过物理预计算表优化，训练速度约 5000 steps/s)
glider_train.py为训练代码 (集成自动评估功能)

# 开发基线
1. 物理计算优化：由于 AoA 固定且 Bank 离散，使用 `physics_table` 预计算平衡状态，避免在 `step` 中调用 `scipy.interpolate`。
2. 状态空间：BANK_BINS=7, W_ACCEL=3, DELTA_W=3。
3. 训练速度：不使用多线程时，通过减少 Python 解释器开销可达到 ~5000 steps/s。
4. 奖励函数：采用混合稠密奖励 `reward = current_uz + 5*w_accel + (dz * lambda)`。其中 `dz * lambda` 为每步高度变化奖励，用于解决长序列中的信用分配(Credit Assignment)问题。
6. 操纵面阻力模拟：为了真实反映频繁改变姿态的代价，引入 `CONTROL_DRAG_MULTIPLIER` (默认1.1)。当 Agent 改变 AoA 或 Bank 索引时，该步物理积分使用 `physics_table_drag` (Cd 放大 1.1x 后的平衡态)，模拟控制面偏转产生的额外诱导阻力。