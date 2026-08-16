# `train_ppo.py`

## 职责

基于 CleanRL 连续动作 PPO 结构训练 `DynamicGliderEnv`。训练使用一个串行环境，避免同时打开多份风场；可使用 GPU。

## 契约

PPO 的网络侧动作通过 Gymnasium `RescaleAction` 转换后才进入环境，因此环境实际控制语义始终是 `[0, 1]^2`。训练观察值按 `config` 中的固定物理量尺度归一化；该归一化不改变传感器定义。奖励不做额外塑形。

训练随机种子、起始帧范围、风场倍率、控制与 PPO 超参数均来自 `config.py`。最终模型、训练 CSV 和图表仅写入项目内的 `q_table/` 与 `trainresult/`。

运行 `python train.py --algo ppo`，短流程可使用 `--steps N`，GPU 可用时默认启用。
