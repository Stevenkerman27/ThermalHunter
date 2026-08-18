"""Normalization contract for the four raw dynamic-environment observations."""

from dataclasses import dataclass
import json
import os

import numpy as np

import config


_OBSERVATION_DIM = 4


@dataclass(frozen=True)
class DynamicObservationNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if mean.shape != (_OBSERVATION_DIM,) or std.shape != (_OBSERVATION_DIM,):
            raise ValueError("dynamic observation normalizer requires four mean and std values")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("dynamic observation normalizer values must be finite")
        if np.any(std <= 0.0):
            raise ValueError("dynamic observation normalizer std values must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def from_samples(cls, observations):
        observations = np.asarray(observations, dtype=np.float64)
        if observations.ndim != 2 or observations.shape[1] != _OBSERVATION_DIM:
            raise ValueError("dynamic observation samples must have shape (n, 4)")
        if len(observations) == 0:
            raise ValueError("dynamic observation samples must not be empty")
        return cls(observations.mean(axis=0), observations.std(axis=0))

    @classmethod
    def from_dict(cls, state):
        if set(state) != {"mean", "std"}:
            raise ValueError("dynamic observation normalizer state must contain only mean and std")
        return cls(state["mean"], state["std"])

    def normalize(self, observations):
        observations = np.asarray(observations, dtype=np.float32)
        if observations.shape[-1:] != (_OBSERVATION_DIM,):
            raise ValueError("dynamic observations must end with four features")
        return ((observations - self.mean) / self.std).astype(np.float32)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def default_dynamic_observation_stats_path():
    return os.path.join(config.TRAIN_RESULT_DIR, "dynamic_observation_normalizer.json")


def save_dynamic_observation_normalizer(normalizer, stats_path=None):
    stats_path = default_dynamic_observation_stats_path() if stats_path is None else stats_path
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as stats_file:
        json.dump(normalizer.to_dict(), stats_file, indent=2)
    return stats_path


def load_dynamic_observation_normalizer(stats_path=None):
    stats_path = default_dynamic_observation_stats_path() if stats_path is None else stats_path
    with open(stats_path, encoding="utf-8") as stats_file:
        return DynamicObservationNormalizer.from_dict(json.load(stats_file))
