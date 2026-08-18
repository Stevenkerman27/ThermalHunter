"""Shared checkpoint persistence for dynamic neural-policy training."""

import os

import pandas as pd
import torch


def checkpoint_model_path(model_path, global_step):
    directory, filename = os.path.split(model_path)
    stem, suffix = os.path.splitext(filename)
    return os.path.join(directory, "checkpoints", f"{stem}_step_{global_step}{suffix}")


def save_training_checkpoint(model, model_path, global_step, csv_rows):
    checkpoint_path = checkpoint_model_path(model_path, global_step)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    for rows, csv_path in csv_rows:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    return checkpoint_path
