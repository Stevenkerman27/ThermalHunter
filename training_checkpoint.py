"""Shared checkpoint persistence for dynamic neural-policy training."""

import os

import pandas as pd
import torch


def checkpoint_model_path(model_path, global_step):
    directory, filename = os.path.split(model_path)
    stem, suffix = os.path.splitext(filename)
    return os.path.join(directory, "checkpoints", f"{stem}_step_{global_step}{suffix}")


def save_model_artifact(model, model_path, metadata):
    if not isinstance(metadata, dict):
        raise ValueError("model artifact metadata must be a dictionary")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, model_path)


def load_model_artifact(model_path, device):
    artifact = torch.load(model_path, map_location=device, weights_only=True)
    if not isinstance(artifact, dict) or set(artifact) != {"model_state_dict", "metadata"}:
        raise ValueError("model artifact must contain model_state_dict and metadata")
    if not isinstance(artifact["metadata"], dict):
        raise ValueError("model artifact metadata must be a dictionary")
    return artifact["model_state_dict"], artifact["metadata"]


def save_training_checkpoint(model, model_path, global_step, csv_rows, metadata):
    checkpoint_path = checkpoint_model_path(model_path, global_step)
    save_model_artifact(model, checkpoint_path, metadata)
    for rows, csv_path in csv_rows:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    return checkpoint_path
