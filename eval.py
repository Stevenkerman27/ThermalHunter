import argparse
import subprocess
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic", action="store_true")
    args, remaining_args = parser.parse_known_args()
    target = "eval_dynamic.py" if args.dynamic else "eval_all.py"
    subprocess.run([sys.executable, target, *remaining_args], check=True)
