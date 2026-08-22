# 训练

统一入口是 `python train.py --algo <name>`，可选 `tabular`、`dqn`、`ppo` 和 `dynamic-dqn`。默认值来自 `config.TRAIN_ALGORITHM`。`--steps N` 对表格 Q 转发为 `--steps`，对其余算法转发为 `--total-timesteps`；`--stats N` 只传给稳态 DQN 的传感器统计阶段，`--cpu` 禁用三个神经网络训练的 CUDA。

每次运行只训练一种算法。动态实验会完整加载风场文件集到内存，因而不得并行启动多个动态训练或动态评估进程。

## 稳态表格 Q

```powershell
python train.py --algo tabular
python glider_train.py --steps 1000
```

表格 Q 使用离散观测环境，Q 表形状直接由环境的 `MultiDiscrete` 观测空间和 9 动作空间确定。训练以 `SEED` 生成起始帧和 epsilon-greedy 随机流；学习率和 epsilon 在总步数的前 90% 线性变化。终止或截断转换不 bootstrap；并列最大 Q 值优先选择中性动作。最终表格和逐回合 `step, episode, return, climb` 日志分别写入 `q_table/` 和 `trainresult/`。

## 稳态 DQN

```powershell
python train.py --algo dqn
python train_dqn.py --total-timesteps 1000
```

训练开始前，`analyze_bins.collect_sensor_stats()` 用同一套起始帧规则重建 `trainresult/sensor_stats.json`；随后创建一个 `SyncVectorEnv`，其数量必须为 1。DQN 的输入是 4 维归一化连续观测，输出 9 个控制动作的 Q 值。训练使用经验回放、epsilon-greedy、MSE TD 损失和目标网络更新；可在 GPU 上运行。

默认最终模型为 `q_table/dqn_model_tuned_v1.pth`，逐回合统计为 `trainresult/dqn_train_stats.csv`，训练曲线为 `trainresult/dqn_train_result.png`，TensorBoard 与阶段检查点写入 `runs/`。参数可显式覆盖模型、CSV、统计和图表路径，供奖励扫描隔离产物。

## 动态 PPO

```powershell
python train.py --algo ppo
python train_ppo.py --total-timesteps 1000
```

动态 PPO 在一个 `DynamicGliderBatchEnv` 中按 `DYNAMIC_NUM_ENVS` 并行推进多个滑翔机，风场文件集在该训练进程内只加载一份。可先独立运行 `python collect_dynamic_observation_stats.py`，生成 `trainresult/dynamic_observation_normalizer.json`；训练会复用该文件，仅在它缺失时顺序运行并关闭一个随机批量环境。实际训练、checkpoint、评估和可视化使用同一份封装在模型工件内的统计量。策略使用 Squashed Gaussian，即对高斯样本施加 `tanh` 并在 log-prob 中加入变量变换修正，动作天然位于 `[-1, 1]^2`，再映射至环境的 `[0, 1]^2` 控制语义。实现使用固定长度 rollout、GAE、裁剪策略目标和未裁剪价值损失；熵系数从 `PPO_ENT_COEF` 线性退火至 `PPO_ENT_COEF_FINAL`。训练控制台每完成 `DYNAMIC_REPORT_EPISODES` 个 episode 打印一次指标。

默认模型、逐回合 CSV 和 TensorBoard 分别为 `q_table/ppo_dynamic_model.pth`、`trainresult/ppo_dynamic_training.csv` 和 `trainresult/ppo_runs/`。另写入 `ppo_dynamic_updates.csv`（loss、entropy、KL、clip fraction、解释方差）。训练过程中不执行策略验证，只按 `DYNAMIC_CHECKPOINT_INTERVAL` 保存 checkpoint；完整评估由独立评估命令执行。

## 动态 DQN

```powershell
python train.py --algo dynamic-dqn
python train_dynamic_dqn.py --total-timesteps 1000
```

动态 DQN 与 PPO 使用同一批量动态环境、同一随机预采样规范的四维观测标准化、奖励和起始帧规则，但通过 `DYNAMIC_DQN_ACTION_LEVELS ** 2` 个离散动作选择速度与滚转命令。每个模型独立封装自己的标准化统计量；实现复用 DQN 的经验回放和 epsilon-greedy 结构，TD 更新使用 Double-DQN 目标和 Huber loss，训练控制台每完成 `DYNAMIC_REPORT_EPISODES` 个 episode 打印一次最近的 TD 指标及该组 episode 的平均回报、平均长度，并写入 `dynamic_dqn_updates.csv`。

默认模型、逐回合 CSV 和 TensorBoard 分别为 `q_table/dynamic_dqn_model.pth`、`trainresult/dynamic_dqn_training.csv` 和 `trainresult/dynamic_dqn_runs/`。训练过程中不执行策略验证，只按 `DYNAMIC_CHECKPOINT_INTERVAL` 保存 checkpoint；完整评估由独立评估命令执行。

动态 DQN 超参数实验以固定评估场景中的 `height_change` 与 `energy_height_change` 分布为筛选依据。训练 CSV 和 TD loss 只用于筛选候选模型；候选模型再用 `python eval.py --dynamic --n 100` 做完整确认。报告必须保留样本数、离散程度和终止原因，且不使用单一高度阈值定义成功。实验允许覆盖默认动态 DQN 模型和日志产物。

## 可复现实验条件

所有训练都使用 `config.SEED`，并从 `[60, 100)` 的同一稳定帧范围随机采样，而不是锁定为同一条轨迹。稳态表格 Q 与稳态 DQN 共享风场、物理参数和奖励 `u_z + w_a * a_z`；唯一的奖励权重是 `w_a`。动态 PPO 与动态 DQN 共享动态环境条件和总能量高度增量奖励。训练期间不执行策略验证；两种动态训练都按 `DYNAMIC_CHECKPOINT_INTERVAL` 保存 checkpoint，训练结束后由独立评估命令验证。逐回合日志包含终止原因和动作统计。动态模型权重及其四维标准化统计量是同一工件，旧的裸 `state_dict` 模型不兼容并会立即报错。
