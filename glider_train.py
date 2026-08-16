import argparse
import csv
import glob
import os
import pickle

import numpy as np

import config
from glider_discrete_simp import GliderEnv


def greedy_tabular_action(action_values):
    best_actions = np.flatnonzero(action_values == np.max(action_values))
    neutral_action = action_values.size // 2
    if neutral_action in best_actions:
        return int(neutral_action)
    return int(best_actions[0])


def train_tabular(total_steps=None, reward_w_accel=None, save_path=None, log_path=None):
    total_steps = config.TABULAR_TOTAL_STEPS if total_steps is None else total_steps
    reward_w_accel = config.REWARD_W_ACCEL_WEIGHT if reward_w_accel is None else reward_w_accel
    save_path = config.SAVE_PATH if save_path is None else save_path
    log_path = os.path.join(config.TRAIN_RESULT_DIR, "tabular_train_stats.csv") if log_path is None else log_path
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if reward_w_accel <= 0:
        raise ValueError("reward_w_accel must be positive")

    h5_files = sorted(
        glob.glob(os.path.join(config.WIND_DIR, "snapshots_s*.h5")),
        key=config.natural_key,
    )
    if not h5_files:
        raise FileNotFoundError(f"no wind files found in {config.WIND_DIR}")

    rng = np.random.default_rng(config.SEED)
    env = GliderEnv(
        h5_file_path=h5_files,
        polar_file_base=config.POLAR_BASE,
        reward_w_accel=reward_w_accel,
        memory_mode=False,
    )
    q_table = np.zeros(tuple(env.observation_space.nvec) + (env.action_space.n,), dtype=np.float32)
    epsilon = config.EPSILON_START
    alpha = config.ALPHA_START
    decay_steps = max(1, int(total_steps * 0.9))
    epsilon_step = (config.EPSILON_START - config.EPSILON_END) / decay_steps
    alpha_step = (config.ALPHA_START - config.ALPHA_END) / decay_steps
    completed_steps = 0
    episode = 0

    try:
        with open(log_path, "w", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(["step", "episode", "return", "climb"])
            while completed_steps < total_steps:
                reset_time = config.sample_start_frame(rng)
                state, info = env.reset(seed=config.SEED + episode, options={"resettime": reset_time})
                initial_height = info["height"]
                episode_return = 0.0
                done = False

                while not done and completed_steps < total_steps:
                    if rng.random() < epsilon:
                        action = int(rng.integers(env.action_space.n))
                    else:
                        action = greedy_tabular_action(q_table[tuple(state)])

                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    td_target = reward if done else reward + config.GAMMA * np.max(q_table[tuple(next_state)])
                    q_table[tuple(state)][action] += alpha * (td_target - q_table[tuple(state)][action])
                    state = next_state
                    episode_return += reward
                    completed_steps += 1
                    epsilon = max(config.EPSILON_END, epsilon - epsilon_step)
                    alpha = max(config.ALPHA_END, alpha - alpha_step)

                writer.writerow([completed_steps, episode, episode_return, info["height"] - initial_height])
                log_file.flush()
                episode += 1
                if episode % 100 == 0:
                    print(
                        f"episode={episode} step={completed_steps}/{total_steps} "
                        f"climb={info['height'] - initial_height:.1f}m epsilon={epsilon:.3f}"
                    )
    finally:
        env.close()

    with open(save_path, "wb") as model_file:
        pickle.dump(q_table, model_file)
    print(f"Tabular training complete: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=config.TABULAR_TOTAL_STEPS)
    parser.add_argument("--w-accel-weight", type=float, default=config.REWARD_W_ACCEL_WEIGHT)
    parser.add_argument("--save-path", default=config.SAVE_PATH)
    parser.add_argument("--log-path", default=os.path.join(config.TRAIN_RESULT_DIR, "tabular_train_stats.csv"))
    args = parser.parse_args()
    train_tabular(args.steps, args.w_accel_weight, args.save_path, args.log_path)
