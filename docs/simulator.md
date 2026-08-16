# `simulator.py`

## 职责

加载一个表格型或 DQN 策略，运行单条轨迹并显示轨迹与飞行量诊断图。命令入口为 `python simulator.py`。

## 使用方式

在 `simulate_with_env()` 内设置 `POLICY_TYPE` 为 `tabular` 或 `dqn`。脚本加载 `q_table/q_table_v0.pkl` 或 `config.DQN_SAVE_PATH`，使用固定风场起始帧 `80`，最多运行 1000 个 RL 步，并以按需读取模式创建环境。

轨迹记录会对水平周期边界进行展开，使 3D 图中的路径连续。绘图包括三维轨迹（按垂直风速着色）、垂直风加速度、翼尖风速差、每步奖励、控制输入与真空速。

脚本内 `CONFIG` 与模型路径选择是可视化专用设置；环境与训练的可调物理参数以 `config.py` 为准。
