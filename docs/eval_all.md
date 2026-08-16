# `eval_all.py`

## 职责

对随机、表格型 Q-learning 和 DQN 三种策略进行同起点、同风场帧的并行对比。命令入口为 `python eval.py`（底层实现为 `eval_all.py`）。

## 工作方式

`MultiGliderEvaluator` 以按需读取模式共享一个 `RBWindField` 和一套预计算气动物理表，为三个滑翔机复制完全相同的初始状态。每个外层时间步先分别计算三种策略的动作，再统一推进一次风场帧，因此策略面对相同风场实现。

脚本默认从 `config.SAVE_PATH` 加载表格型策略，从 `config.DQN_SAVE_PATH` 加载 DQN 权重，并从 `sensor_stats.json` 获取 DQN 归一化参数。`--tabular-model`、`--dqn-model`、`--sensor-stats`、`--w-accel-weight`、`--output-csv` 和 `--output-plot` 可以显式指向一组实验产物。任一模型或统计缺失时立即报错。

结果以最终高度减初始高度计为爬升。评估以固定种子按训练相同规则随机生成本轮起始帧，输出三策略的均值和标准差，并保存 `trainresult/compare_eval_result.png` 与逐 episode 的 `trainresult/evaluation_episodes.csv`。

每个 episode 的初始状态和随机策略动作使用独立、由评估种子派生的随机流，避免任一策略的终止时机影响后续评估场景。

评估不被训练脚本隐式调用。
