# `reward_sweep.py`

## 职责

对垂直风速度与垂直风加速度奖励的相对权重进行可复现比较。基准系数固定为 `u_z:1`，权重列表只由 `config.REWARD_W_ACCEL_SWEEP_WEIGHTS` 定义。

## 流程

`python reward_sweep.py` 串行执行 `1:1`、`1:3`、`1:5` 三组实验。每组依次训练表格 Q 和 DQN，两个训练进程不并发；每个模型、训练日志、DQN 传感器统计与评估结果都按权重单独保存。

所有模型在以 `EVAL_SEED` 产生的同一批场景上评估。输出 `trainresult/reward_sweep_evaluation.csv`、`trainresult/reward_sweep_summary.csv` 和 `trainresult/reward_sweep_evaluation.png`。

## 参数

`--steps N` 为每个算法、每个权重的训练步数；`--stats N` 为每次 DQN 训练的传感器统计回合数；`--cpu` 禁用 DQN 的 GPU。
