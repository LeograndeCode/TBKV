#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[2]
    eventful_root = repo_root / "eventful-transformer"
    target_script = eventful_root / "scripts" / "evaluate" / "vivit_kinetics400.py"

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    local_path = ".:.."
    env["PYTHONPATH"] = f"{local_path}:{existing}" if existing else local_path

    cmd = [sys.executable, str(target_script)] + sys.argv[1:]
    completed = subprocess.run(cmd, cwd=eventful_root, env=env)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
