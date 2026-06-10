# DQN Hyperparameter Tuning Log

| Attempt | Date | LR | GAMMA | EPSILON_START/END | TIMESTEPS | Avg Climb (m) | Max Climb (m) | Notes |
|---------|------|----|-------|-------------------|-----------|---------------|---------------|-------|
| 0 (Baseline) | 2026-06-02 | 1e-4 | 0.99 | 1.0 / 0.05 | 200000 | 152.2 | 871.3 | Baseline with current config.py parameters. |
| 1 | 2026-06-02 | 5e-5 | 0.99 | 1.0 / 0.05 | 600000 | 180.0 | 948.3 | Target Reached! Batch size increased to 128. Std decreased significantly (221.6m). Stable training. |
| 2 | 2026-06-02 | 8e-5 | 0.995 | 1.0 / 0.05 | 400000 | 129.4 | 827.1 | Performance regressed. Higher LR and fewer steps led to premature/poor convergence. Stability (219.2m) is still good. |
| 3 | 2026-06-02 | 6e-5 | 0.995 | 1.0 / 0.05 | 500000 | 213.9 | 1004.5 | Breakthrough! Exceeded 213m. Best stability (211.4m). Balanced LR and steps worked perfectly with high Gamma. |
