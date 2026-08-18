# 评估

训练不会隐式执行评估。两种评估入口均以 `EVAL_SEED` 生成可复现的随机场景，并仅比较同一环境中的策略。

## 稳态策略比较

```powershell
python eval.py
python eval_all.py --tabular-model path --dqn-model path --sensor-stats path
```

默认比较随机策略、表格 Q 和稳态 DQN。`MultiGliderEvaluator` 为三者复制同一初始状态，在每个外层步骤收集三个动作、分别推进飞行器后只推进一次共享风场，所以每种策略经历相同的风场帧序列。每个 episode 的初始状态和随机动作各自使用由评估种子派生的随机流。

表格模型、DQN 权重和 `trainresult/sensor_stats.json` 均为必需输入，缺失时立即报错。结果是最终高度减去初始高度，逐场景 CSV 默认写入 `trainresult/evaluation_episodes.csv`，比较图默认写入 `trainresult/compare_eval_result.png`。`--w-accel-weight` 必须与所评估的稳态模型奖励配置一致。

## 动态策略比较

```powershell
python eval.py --dynamic
python eval_dynamic.py --n 10 --model path --dqn-model path
```

动态评估在相同的起始帧、位置、高度和航向下依次比较 `Random grid`、`Cruise`、连续 PPO 与动态 DQN。随机网格和巡航基线使用离散动作包装器；巡航固定命令 `DYNAMIC_BASELINE_SPEED_ACTION` 与 `DYNAMIC_BASELINE_ROLL_ACTION`。PPO 采用 `tanh(mean)` 的确定性有界动作，DQN 采用最大 Q 动作。

入口需要 PPO 和动态 DQN 两个模型都存在，且每个模型工件都必须包含其训练时的四维观测标准化统计量。评估开始时只加载一份完整风场数据到内存，策略环境顺序借用该只读资源，完成后统一关闭。输出每个策略和场景的回报、步数、高度变化、总能量高度变化和终止原因到 `trainresult/dynamic_evaluation.csv`，并生成 `trainresult/dynamic_evaluation.png`；图中显示两个指标的箱线、原始点、均值、标准差、中位数和样本数。

动态结果不能与稳态表格 Q 或稳态 DQN 的数值直接作为算法优劣比较，因为环境动力学和奖励不同。
