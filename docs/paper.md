# Workshop Paper Scope

The approved workshop-paper claim is energy-harvesting soaring in time-varying turbulent wind, not route navigation, safety certification, real-flight deployment, or a completed sim-to-real transfer.

The shared wind input is DNS Rayleigh--Benard convection at `Ra = 5e7`. The paper may describe it as a controlled numerical experiment and must identify it as simulation data.

The paper compares policies only within the same environment. The steady discrete and dynamic aerodynamic environments are complementary fidelity regimes, not a cross-regime algorithm ranking.

Primary outcomes are matched-scenario height change and total-energy-height change. Reports must include sample count, distributional uncertainty, and termination reasons. No fixed `100 m` success threshold is used.

The steady experiment may use DQN decision boundaries to select transparent Tabular-Q sensor bins. This is a representation-design experiment: the bin-selection procedure, training budget, seeds, and held-out evaluation protocol must be recorded before results are compared.

## Draft source

`paper/thermal_hunter.tex` is the blinded NeurIPS workshop manuscript source.
It covers the abstract, motivation, related work, both environment setups,
protocol, steady-regime results, held-out dynamic results, discussion and
limitations, and conclusion.

## Dynamic Completion Protocol

The dynamic regime is the higher-fidelity progression of the same local
energy-harvesting task. It keeps the DNS input, wind scaling, random scenario
range, and matched-scenario evaluation rule, but replaces quasi-steady lookup
dynamics with RK4-integrated motion, time-interpolated wind, actuator dynamics,
and total-energy-height reward.

PPO uses continuous commands and dynamic DQN uses a 3-by-3 command grid. Their
comparison is therefore a controller-variant comparison, not a pure algorithm
ranking. Dynamic results must report height change, total-energy-height change,
episode length, and all termination reasons. Upper-altitude termination is
evidence that the episode ended after gain, not a sustained-soaring or safety
claim.

Paper-level dynamic estimates use three independent training seeds `(11, 22,
33)` and a new 100-scenario evaluation suite generated with seed `20260821`.
The resulting replicas are stored under `trainresult/dynamic_multiseed/models/`.
Training seed is the replication unit; scenario distributions remain within-seed
evidence. Scenario bootstrap intervals and between-seed standard deviations are
reported separately.
