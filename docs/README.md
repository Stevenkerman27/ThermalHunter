# ThermalHunter 文档

本目录描述主滑翔机工作流；不覆盖 `practice/` 中独立实验，也不把默认运行 `CartPole-v1` 的 `dqn.py` 视为滑翔机训练入口。项目研究目标是让滑翔机在时变湍流风场中获取能量，不把当前任务表述为航路导航或真实飞行验证。

## 功能地图

- [环境](environment.md)：输入风场、极线、稳态与动态 Gymnasium 环境的物理和接口契约。
- [训练](training.md)：表格 Q、稳态 DQN、动态 PPO、动态 DQN 的入口、共同实验条件与产物。
- [评估](evaluation.md)：固定随机场景下的稳态与动态策略比较。
- [配置](config.md)：`config.py` 集中定义的可调共享参数。
- [数据与产物](data.md)：HDF5、极线、传感器统计、模型和结果文件。
- [工具](tools.md)：奖励权重扫描、单轨迹仿真、统计和历史工具。
- [论文范围](paper.md)：当前 workshop 论文的已批准主张和实验边界。

## 入口

从项目根目录运行：

```powershell
python train.py --algo tabular
python train.py --algo dqn
python train.py --algo ppo
python train.py --algo dynamic-dqn
python eval.py
python eval.py --dynamic
```

稳态和动态实验的物理模型、观测、动作和奖励不同，只在各自环境内比较算法。所有共享配置只在 `config.py` 中定义。
