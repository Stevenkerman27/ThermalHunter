# DQN Training Print Cycle Update Design

## Objective
Change the terminal output in `train_dqn.py` from printing every single episode's climb height and return to printing the average climb height and average return every 10 episodes. This reduces terminal spam while providing a smoother moving average of training progress.

## Approach
1. **State Tracking**: Add two lists, `recent_returns` and `recent_heights`, before the main training loop (`global_step` loop).
2. **Data Collection**: When an episode completes (`if "_episode" in infos` and `d` is true), append the episode's return and final height to these lists.
3. **Periodic Printing**: Check if the length of `recent_heights` reaches `10`. If so, calculate the mean of both lists, print them to the terminal (e.g., `global_step=..., avg_return (last 10 eps)=..., avg_height=...`), and then clear both lists.
4. **TensorBoard Logging**: The tensorboard logging (`writer.add_scalar`) will remain unchanged, continuing to log individual episode data so metrics charts are still granular.

## Trade-offs & Scope
- **Trade-off**: The terminal output will be delayed by 9 episodes before showing up, but it will be much cleaner.
- **Scope**: Modifies `train_dqn.py` only. No changes to the environment or other tracking mechanisms.