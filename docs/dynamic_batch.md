# Dynamic Batch Execution

`DynamicGliderBatchEnv` is the synchronous vector interface for dynamic PPO,
dynamic DQN, and dynamic evaluation. It owns, or borrows, exactly one read-only
`RBWindField`; every trajectory shares it in one process. Its state, RK4
integration, and wind interpolation are vectorized over the leading environment
dimension.

`reset()` returns raw observations with shape `(num_envs, 4)`. `step()` returns
batched observations, rewards, termination flags, truncation flags, and
per-environment info. Training enables automatic reset for ended slots. Batch
evaluation and trajectory collection disable automatic reset so terminal states
remain available.

`config.DYNAMIC_NUM_ENVS` is the only default environment-count definition.
PPO interprets `PPO_NUM_STEPS` per environment; the defaults retain 1,024 total
transitions per update. DQN collects one transition per active environment and
executes every scheduled update crossed by the accumulated environment-step
count, retaining the original update ratio.

Training does not run periodic policy evaluation. Every
`config.DYNAMIC_CHECKPOINT_INTERVAL` accumulated environment steps, both
trainers save a step-labelled model checkpoint and flush their cumulative
training CSV rows to the configured result paths. Final model and CSV saves
remain unchanged.
