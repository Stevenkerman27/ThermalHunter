# RL-soar: Autonomous Glider Soaring via Reinforcement Learning

RL-soar is a Reinforcement Learning (RL) framework for training autonomous gliders to harvest energy from spatio-temporal wind fields, such as thermals and updrafts.

## Overview

The project implements a 3D simulation environment where agents learn control policies for Angle of Attack and Bank angle based on local sensor data, including vertical acceleration and wingtip velocity differentials.

## Technical Features

- **Optimized Simulator**: The `GliderEnv` utilizes pre-calculated physics tables to achieve training throughput of approximately 5,000 steps per second.
- **DQN Implementation**: A reproducible Deep Q-Network with a single-environment replay buffer and fixed-seed evaluation.
- **Physics Modeling**:
    - Aerodynamic polar curves for glider performance.
    - Control surface drag modeling, simulating additional drag during attitude transitions.
    - Periodic boundary conditions for continuous flight in large-scale wind fields.
- **Reward System**: The steady environment uses vertical wind velocity and acceleration; the dynamic PPO environment uses total-energy height change.
- **Benchmarking Suite**: Parallel evaluation of multiple agents (Random, Tabular, and DQN) within identical wind field realizations for comparative analysis.
- **Analysis Tools**: Utilities for 3D trajectory visualization, wind field analysis, and sensor distribution diagnostics.

## Project Structure

- `docs/`: Chinese module definitions for the current glider workflow.
- `glider_discrete_simp.py`: Core Gymnasium-compatible environment.
- `train.py`: Unified training entry; selects tabular Q-learning or DQN.
- `train_dqn.py`: DQN training implementation.
- `glider_dynamic.py`: Non-steady point-mass glider environment for PPO.
- `train_ppo.py`: CleanRL-style continuous-action PPO training.
- `config.py`: Centralized configuration for physics, environment, and training parameters.
- `simulator.py`: Interactive visualization tool for trajectories and manual flight.
- `eval.py`: Fixed-scenario evaluation entry.
- `glider_train.py`: Tabular Q-learning implementation.
- `plot_dqn_slice.py` / `plot_dqn_train.py`: Policy analysis and training progress visualization.
- `wind/`: Directory for HDF5 wind field data.

## Installation and Setup

### Prerequisites
- Python 3.x (Environment recommendation: `conda create -n myml python=3.10`)
- PyTorch
- Gymnasium
- h5py
- Matplotlib, NumPy, Pandas

### Data Preparation
Place HDF5 wind field snapshots in the `wind/` directory. Files must follow the `snapshots_s*.h5` naming convention.

## Usage

### Training
Execute tabular Q-learning with the configured default:
```bash
python train.py
```

Select DQN explicitly:
```bash
python train.py --algo dqn
```

Train the dynamic PPO experiment:
```bash
python train.py --algo ppo
```
Logs and model checkpoints are stored under `trainresult/`, `runs/`, and `q_table/`.

### Policy Evaluation
Compare trained models against baseline policies:
```bash
python eval.py
```

Evaluate the dynamic PPO experiment:
```bash
python eval.py --dynamic
```

### Trajectory Visualization
Visualize the flight path of a specific policy:
```bash
python simulator.py
```
Configure `POLICY_TYPE` and model paths within the script before execution.
