# 滑翔机模块文档

本文档仅覆盖主滑翔机工作流，不覆盖 `practice/` 下的独立强化学习实验，也不将默认运行 `CartPole-v1` 的 `dqn.py` 视为滑翔机模块。

## 主链路

`wind/snapshots_s*.h5` + `glider.polar` -> `glider_discrete_simp.py` -> `train.py` -> `eval.py` / `simulator.py` / 分析脚本。

## 模块

- [配置](config.md)：唯一的滑翔机全局可调参数来源。
- [数据契约](data.md)：风场、极线、传感器统计和产物格式。
- [环境](glider_discrete_simp.md)：风场读取、稳态气动与 Gymnasium 环境。
- [动态环境](glider_dynamic.md)：实时空速动力学、连续控制与物理传感器。
- [PPO 训练](train_ppo.md)：动态环境的 CleanRL 连续动作 PPO。
- [动态评估](eval_dynamic.md)：随机、定速定滚转与 PPO 的固定场景比较。
- [DQN 训练](train_dqn.md)：连续传感器观测的 DQN。
- [表格型训练](glider_train.md)：离散观测的 Q-learning。
- [统一评估](eval_all.md)：随机、表格型和 DQN 策略在固定种子生成的随机同起点场景中对比。
- [奖励权重扫描](reward_sweep.md)：串行训练 `1:1`、`1:3`、`1:5` 的表格 Q 和 DQN，并汇总固定场景评估。
- [轨迹仿真](simulator.md)：单一策略的交互式结果绘制。
- [分析工具](analysis_tools.md)：传感器统计、模型、策略和风场可视化。
- [旧策略生成器](make_pkl.md)：与当前分箱契约不兼容的历史工具。

## 运行环境

使用项目约定的 `myml` Conda 环境。默认训练为 `python train.py`；`python train.py --algo dqn` 选择 DQN，`python eval.py` 执行固定种子随机评估。所有脚本均从项目根目录启动，以保证相对路径和生成物位置正确。
