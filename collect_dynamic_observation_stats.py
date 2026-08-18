"""Generate the shared dynamic-observation normalization statistics."""

import config
from dynamic_observation import save_dynamic_observation_normalizer
from train_ppo import collect_dynamic_observation_normalizer, dynamic_wind_paths


def main():
    normalizer = collect_dynamic_observation_normalizer(dynamic_wind_paths(), config.SEED)
    stats_path = save_dynamic_observation_normalizer(normalizer)
    print(f"Saved dynamic observation normalization to {stats_path}")
    print(f"mean={normalizer.mean.tolist()}")
    print(f"std={normalizer.std.tolist()}")


if __name__ == "__main__":
    main()
