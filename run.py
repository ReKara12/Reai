"""Convenience runner script for Reflex-Agent."""

import os
import sys
import subprocess
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def main():
    root = Path(__file__).resolve().parent
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print("Reflex-Agent Operations Runner:")
        print("  python run.py test        - Run complete pytest test suite")
        print("  python run.py benchmark   - Run System 1 latency benchmark")
        print("  python run.py demo        - Run interactive Tkinter GUI demo")
        print("  python run.py voice       - Run 100% local voice assistant (Push-to-Talk)")
        print("  python run.py hud [args]  - Launch Andy Gao-style Floating Glassmorphic HUD")
        print("  python run.py train [args]- Run SFT fine-tuning on GPU overnight (silent, no UI)")
        print("  python run.py cli [args]  - Forward arguments to src.main")
        return

    try:
        cmd = args[0]
        if cmd == "test":
            sys.exit(subprocess.call([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=str(root)))
        elif cmd == "benchmark":
            sys.exit(subprocess.call([sys.executable, "-m", "src.main", "--benchmark", "--mock"], cwd=str(root)))
        elif cmd == "demo":
            sys.exit(subprocess.call([sys.executable, "run_demo.py"], cwd=str(root)))
        elif cmd == "hud":
            sys.exit(subprocess.call([sys.executable, "-m", "src.ui.floating_hud"] + args[1:], cwd=str(root)))
        elif cmd == "voice":
            sys.exit(subprocess.call([sys.executable, "-m", "src.main", "--voice"] + args[1:], cwd=str(root)))
        elif cmd == "train":
            train_data_path = root / "data" / "train_os_reflex.json"
            if not train_data_path.exists():
                print("Generating synthetic OS reflex dataset...")
                subprocess.run([sys.executable, "src/training/dataset_generator.py"], cwd=str(root), check=True)
            sys.exit(subprocess.call([sys.executable, "src/training/train_laya.py"] + args[1:], cwd=str(root)))
        elif cmd == "cli":
            sys.exit(subprocess.call([sys.executable, "-m", "src.main"] + args[1:], cwd=str(root)))
        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nİşlem kullanıcı tarafından durduruldu.")
        sys.exit(0)


if __name__ == "__main__":
    main()
