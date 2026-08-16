import argparse
import subprocess
import sys

import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("tabular", "dqn", "ppo"), default=config.TRAIN_ALGORITHM)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stats", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.algo == "tabular":
        command = [sys.executable, "glider_train.py"]
    elif args.algo == "dqn":
        command = [sys.executable, "train_dqn.py"]
    else:
        command = [sys.executable, "train_ppo.py"]
    if args.steps is not None:
        command.extend(["--steps" if args.algo == "tabular" else "--total-timesteps", str(args.steps)])
    if args.stats is not None and args.algo == "dqn":
        command.extend(["--sensor-stats-episodes", str(args.stats)])
    if args.cpu and args.algo in ("dqn", "ppo"):
        command.append("--no-cuda")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
