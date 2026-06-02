# Hyperparameter Tuning Log

| Attempt | Date | ALPHA | GAMMA | EPSILON_START/END | EPISODES | Avg Climb Height (m) | Max Climb Height (m) | Notes |
|---------|------|-------|-------|-------------------|----------|----------------------|----------------------|-------|
| 0 (Baseline) | 2026-06-01 | 0.04 | 0.999 | 1.0 / 0.01 | 8000 | -39.4 | TBD | Initial parameters from config.py |
| 1 | 2026-06-01 | 0.01 | 0.99 | 1.0 / 0.05 | 10000 | 74.3 | TBD | Significant improvement. Convergence visible (R~800). Short of 100m. |
| 2 | 2026-06-01 | 0.002 | 0.99 | 1.0 / 0.02 | 10000 | -8.6 | TBD | Performance regressed. Low ALPHA was too conservative. |
| 3 | 2026-06-01 | 0.02->0.001 | 0.99 | 1.0 / 0.02 | 10000 | 66.2 | TBD | Used Alpha decay & Lambda=0.4. High peaks but inconsistent final eval. |
| 4 | 2026-06-01 | 0.005 | 0.99 | 1.0 / 0.01 | 10000 | 93.9 | TBD | Moderate ALPHA & Lambda=0.3. Very stable, nearly hit 100m. |
| 5 | 2026-06-01 | 0.005 | 0.98 | 1.0 / 0.01 | 10000 | 103.5 | TBD | Final Sprint: Lambda=0.5, Gamma=0.98. Target Achieved! |
