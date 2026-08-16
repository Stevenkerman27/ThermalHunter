# `glider_discrete_simp.py`

## 职责

实现主仿真层：HDF5 风场访问、基于极线的稳态气动计算，以及 Gymnasium 兼容的离散控制环境。

## 公开组件

- `RBWindField(h5_paths, domain_size, memory_mode)`：串联风场文件；`get_wind(x, y, z)` 对 `ux/uy/uz` 进行三线性插值；`vector_rms()` 统计原始三维速度 RMS；`reset(t_index)` 和 `step_time()` 管理全局时间帧；`close()` 释放 HDF5 资源。
- `GliderPhysics(polar_file_base)`：从极线插值升阻系数；`get_steady_state(alpha_rad, bank_rad, drag_mult)` 返回真空速、下滑角和航向角速度。
- `GliderEnv(...)`：9 个控制动作的 Gymnasium 环境。

## 环境契约

动作 `0..8` 解码为 `(AoA delta, Bank delta)`，每一维均为 `-1/0/+1`。执行控制改变时使用带额外阻力的预计算物理表。

`continuous_obs=False` 返回整型 `[aoa_idx, bank_idx, w_accel_bin, delta_w_bin]`，后两项通过 `config` 分箱和迟滞生成；这是表格型 Q 表的索引。`continuous_obs=True` 返回 float32 `[aoa_idx, bank_idx, w_accel, delta_w]`，供 DQN 使用。

每个 RL 步执行 `n_phys_per_rl` 次积分。奖励为当前垂直风速和垂直风加速的加权和：`u_z + w_accel_weight * a_z`。高度到达域高的 10% 或 90% 时 `terminated`；风场帧耗尽时 `truncated`。`info` 提供传感器、控制量、高度、真空速与垂直风速。

默认按需读取风场。环境以中位迎角、零滚转稳态空速和 `config.WIND_RMS_TO_TAS_RATIO` 推导风场倍率，只改变速度幅值，不改变时间推进。调用方必须调用 `close()`。
