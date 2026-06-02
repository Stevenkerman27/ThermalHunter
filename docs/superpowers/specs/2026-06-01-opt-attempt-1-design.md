# Design Document: Hyperparameter Optimization Attempt 1

**Date:** 2026-06-01
**Topic:** Improving Tabular Q-Learning performance for glider soaring.
**Goal:** Achieve average climb height > 100m.

## Current Baseline
*   **Mean Climb:** ~ -39.4m (Training Script Eval)
*   **Observations:** High variance in Q-values, inconsistent climb performance despite DQN achieving 200m+.

## Proposed Changes
| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `ALPHA` | 0.04 | 0.01 | Reduce noise impact from stochastic wind field. |
| `GAMMA` | 0.999 | 0.99 | Focus on medium-term soaring decisions (3-5 mins) vs 15+ mins. |
| `EPISODES` | 8000 | 10000 | Allow more time for low-alpha learning to stabilize. |
| `EPSILON_END` | 0.01 | 0.05 | Maintain exploration in the neighborhood of learned policies. |
| `CONTROL_DRAG_MULTIPLIER` | 1.2 | 1.1 | Reduce penalty for attitude changes to encourage active soaring. |

## Strategy
1.  Apply changes to `config.py`.
2.  Execute `glider_train.py` for 10,000 episodes.
3.  Observe internal evaluation results (Mean/Max Climb).
4.  Update `tuning_log.md` with results and analysis.

## Success Criteria
*   Mean Climb Height > 0m (improving from baseline).
*   Q-value curves in `train_result.png` showing smoother convergence.
