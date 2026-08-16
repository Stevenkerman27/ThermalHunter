# `glider_train.py`

## 职责

以离散传感器观测训练表格型 Q-learning 策略。`train_tabular()` 可被训练入口调用；`python glider_train.py --steps N` 可单独运行。

## 流程

收集全部 `snapshots_s*.h5`，以按需读取模式创建默认离散观测的 `GliderEnv`。Q 表形状由环境的 `MultiDiscrete` 观测空间和 9 动作空间动态确定。

每个 episode 从训练与评估共用的随机帧范围采样起点，范围避开风场初始瞬态；使用配置种子和 epsilon-greedy 选动作。训练以总步数限制，终止与风场耗尽截断状态均不 bootstrap；epsilon 与学习率在前 90% 步数线性衰减。

表格 Q 的并列最大动作优先选中性动作，避免未访问状态因数组索引顺序而被解释为固定方向的操纵。

表格 Q 与 DQN 均使用环境返回的同一奖励：垂直风速度加上由 `w_accel_weight` 指定的垂直风加速度加权值。不包含高度变化或生存项；两种算法只在值函数表示与优化方法上不同。

## 产物与后处理

最终表写入 `config.SAVE_PATH`，逐 episode 训练数据写入 `trainresult/tabular_train_stats.csv`。训练不隐式评估；统一比较由 `eval.py` 负责。
