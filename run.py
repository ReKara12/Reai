"""Convenience runner script for Reflex-Agent."""

import sys
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print("Reflex-Agent Operations Runner:")
        print("  python run.py test        - Run complete pytest test suite")
        print("  python run.py benchmark   - Run System 1 latency benchmark")
        print("  python run.py demo        - Run interactive Tkinter GUI demo")
        print("  python run.py voice       - Run 100% local voice assistant (Push-to-Talk)")
        print("  python run.py cli [args]  - Forward arguments to src.main")
        return

    cmd = args[0]
    if cmd == "test":
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=str(root)))
    elif cmd == "benchmark":
        sys.exit(subprocess.call([sys.executable, "-m", "src.main", "--benchmark", "--mock"], cwd=str(root)))
    elif cmd == "demo":
        sys.exit(subprocess.call([sys.executable, "run_demo.py"], cwd=str(root)))
    elif cmd == "voice":
        sys.exit(subprocess.call([sys.executable, "-m", "src.main", "--voice"] + args[1:], cwd=str(root)))
    elif cmd == "cli":
        sys.exit(subprocess.call([sys.executable, "-m", "src.main"] + args[1:], cwd=str(root)))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
