# DQN Print Cycle Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify `train_dqn.py` to calculate and print the average return and climb height every 10 episodes instead of printing every single episode, reducing terminal spam.

**Architecture:** We will maintain two lists `recent_returns` and `recent_heights` to accumulate the metric values from `infos["episode"]["r"]` and `infos["height"]`. When the list size reaches 10, we compute the means, print them, and clear the lists. This logic will replace the existing per-episode `print` statement inside the `if d:` block.

**Tech Stack:** Python, Numpy

---

### Task 1: Update Print Logic in `train_dqn.py`

**Files:**
- Modify: `train_dqn.py`

- [ ] **Step 1: Initialize accumulators**

In `train_dqn.py`, locate the start of the game loop:
```python
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
```
Add initialization for the accumulators right before the loop:
```python
    recent_returns = []
    recent_heights = []

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
```

- [ ] **Step 2: Update the printing logic**

Locate the episode logging section inside the loop:
```python
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "_episode" in infos:
            for idx, d in enumerate(infos["_episode"]):
                if d:
                    print(f"global_step={global_step}, episodic_return={infos['episode']['r'][idx]:.2f}, height={infos['height'][idx]:.1f}")
                    writer.add_scalar("charts/episodic_return", infos["episode"]["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", infos["episode"]["l"][idx], global_step)
                    writer.add_scalar("charts/final_height", infos["height"][idx], global_step)
```

Replace the single `print` statement with accumulation and batch printing:
```python
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "_episode" in infos:
            for idx, d in enumerate(infos["_episode"]):
                if d:
                    recent_returns.append(infos["episode"]["r"][idx])
                    recent_heights.append(infos["height"][idx])

                    writer.add_scalar("charts/episodic_return", infos["episode"]["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", infos["episode"]["l"][idx], global_step)
                    writer.add_scalar("charts/final_height", infos["height"][idx], global_step)

                    if len(recent_heights) >= 10:
                        avg_return = np.mean(recent_returns)
                        avg_height = np.mean(recent_heights)
                        print(f"global_step={global_step}, avg_return (last 10 eps)={avg_return:.2f}, avg_height={avg_height:.1f}")
                        recent_returns.clear()
                        recent_heights.clear()
```

- [ ] **Step 3: Run the code to verify**

Run the script to ensure it doesn't crash and prints every 10 episodes:
Run: `python train_dqn.py --total_timesteps 2000` (or another small number)
Expected: The script should run and you should see `global_step=..., avg_return (last 10 eps)=..., avg_height=...` outputted in the terminal.

- [ ] **Step 4: Commit the changes**

```bash
git add train_dqn.py
git commit -m "feat: batch terminal output to print average metrics every 10 episodes"
```