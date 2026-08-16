import glob
import json
import os

import numpy as np

import config
from glider_discrete_simp import GliderEnv


def collect_sensor_stats(episodes=None, stats_path=None):
    episodes = config.SENSOR_STATS_EPISODES if episodes is None else episodes
    stats_path = os.path.join(config.BASE_DIR, "sensor_stats.json") if stats_path is None else stats_path
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    h5_files = sorted(
        glob.glob(os.path.join(config.WIND_DIR, "snapshots_s*.h5")),
        key=config.natural_key,
    )
    if not h5_files:
        raise FileNotFoundError(f"no wind files found in {config.WIND_DIR}")

    rng = np.random.default_rng(config.SEED)
    env = GliderEnv(h5_file_path=h5_files, polar_file_base=config.POLAR_BASE, memory_mode=False)
    accel_values = []
    delta_values = []
    try:
        for episode in range(episodes):
            reset_time = config.sample_start_frame(rng)
            _, _ = env.reset(seed=config.SEED + episode, options={"resettime": reset_time})
            done = False
            while not done:
                _, _, terminated, truncated, info = env.step(int(rng.integers(env.action_space.n)))
                accel_values.append(info["w_accel"])
                delta_values.append(info["delta_w"])
                done = terminated or truncated
    finally:
        env.close()

    stats = {
        "w_accel": {"mean": float(np.mean(accel_values)), "std": float(np.std(accel_values))},
        "delta_w": {"mean": float(np.mean(delta_values)), "std": float(np.std(delta_values))},
    }
    if stats["w_accel"]["std"] <= 0.0 or stats["delta_w"]["std"] <= 0.0:
        raise ValueError("sensor statistics must have non-zero standard deviation")
    with open(stats_path, "w", encoding="utf-8") as stats_file:
        json.dump(stats, stats_file, indent=2)
    print(f"Sensor statistics saved to {stats_path}")
    return stats


if __name__ == "__main__":
    collect_sensor_stats()
