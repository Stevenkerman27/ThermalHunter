# ThermalHunter

用于研究滑翔机如何利用时空风场获取能量的强化学习项目。项目包含两套互不混用的仿真与评估链路：基于稳态极线的离散控制环境，以及基于实时气动力积分的动态环境。

完整的中文说明在 [docs/README.md](docs/README.md)。所有命令应从项目根目录、使用约定的 `myml` Conda 环境运行。

## 快速运行

训练一项算法：

```powershell
python train.py
python train.py --algo dqn
python train.py --algo ppo
python train.py --algo dynamic-dqn
```

评估对应的策略组：

```powershell
python eval.py
python eval.py --dynamic
```

`train.py` 一次只启动一种算法。动态 PPO、动态 DQN 和动态评估都会把完整风场文件集读入内存，因此不要并行运行多个动态训练或评估进程。

## 文档

- [环境](docs/environment.md)：风场访问、稳态和动态飞行模型、观测、动作、奖励与终止。
- [训练](docs/training.md)：表格 Q、稳态 DQN、动态 PPO、动态 DQN。
- [评估](docs/evaluation.md)：两套评估流程及公平比较条件。
- [配置](docs/config.md)：`config.py` 中唯一的共享可调参数来源。
- [数据与产物](docs/data.md)：输入数据、模型与结果格式。
- [工具](docs/tools.md)：奖励扫描、轨迹和分析工具。
