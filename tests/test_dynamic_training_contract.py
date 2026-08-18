import numpy as np
import pytest
import torch

from dynamic_observation import DynamicObservationNormalizer
from train_ppo import annealed_entropy_coefficient, load_or_collect_dynamic_observation_normalizer
from training_checkpoint import load_model_artifact, save_model_artifact


def test_dynamic_normalizer_standardizes_each_existing_feature():
    samples = np.array(
        [[100.0, -2.0, -0.5, -0.2], [300.0, 2.0, 0.5, 0.2]], dtype=np.float32
    )

    normalizer = DynamicObservationNormalizer.from_samples(samples)

    np.testing.assert_allclose(normalizer.normalize(samples), [[-1.0] * 4, [1.0] * 4])
    assert DynamicObservationNormalizer.from_dict(normalizer.to_dict()).to_dict() == normalizer.to_dict()


def test_dynamic_normalizer_rejects_nonpositive_feature_std():
    with pytest.raises(ValueError, match="std values must be positive"):
        DynamicObservationNormalizer.from_samples(np.ones((2, 4), dtype=np.float32))


def test_dynamic_model_artifact_persists_normalization_contract(monkeypatch):
    model = torch.nn.Linear(4, 2)
    normalizer = DynamicObservationNormalizer(np.zeros(4), np.ones(4))
    saved_artifacts = {}

    monkeypatch.setattr("training_checkpoint.os.makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "training_checkpoint.torch.save",
        lambda artifact, path: saved_artifacts.setdefault(path, artifact),
    )
    monkeypatch.setattr(
        "training_checkpoint.torch.load",
        lambda path, map_location, weights_only: saved_artifacts[path],
    )

    save_model_artifact(model, "model.pth", {"observation_normalizer": normalizer.to_dict()})
    state_dict, metadata = load_model_artifact("model.pth", "cpu")

    assert set(state_dict) == {"weight", "bias"}
    assert DynamicObservationNormalizer.from_dict(metadata["observation_normalizer"]).to_dict() == normalizer.to_dict()


def test_ppo_entropy_schedule_reaches_zero_after_exploration():
    assert annealed_entropy_coefficient(0.01, 1, 5) == pytest.approx(0.01)
    assert annealed_entropy_coefficient(0.01, 5, 5) == pytest.approx(0.0)


def test_dynamic_training_reuses_existing_normalizer(monkeypatch):
    normalizer = DynamicObservationNormalizer(np.zeros(4), np.ones(4))

    monkeypatch.setattr("train_ppo.default_dynamic_observation_stats_path", lambda: "stats.json")
    monkeypatch.setattr("train_ppo.os.path.isfile", lambda path: True)
    monkeypatch.setattr("train_ppo.load_dynamic_observation_normalizer", lambda path: normalizer)
    monkeypatch.setattr(
        "train_ppo.collect_dynamic_observation_normalizer",
        lambda *args: pytest.fail("existing normalization stats must not trigger sampling"),
    )

    assert load_or_collect_dynamic_observation_normalizer(["wind.h5"], 1) is normalizer
