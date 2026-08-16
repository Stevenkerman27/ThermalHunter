import subprocess
import sys


if __name__ == "__main__":
    subprocess.run([sys.executable, "eval_all.py"], check=True)
