import os
import platform
import subprocess
from datetime import datetime
from typing import Any, Dict

def _try(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""

def collect_provenance() -> Dict[str, Any]:
    env = {}
    for k, v in os.environ.items():
        if any(x in k.upper() for x in ["TOKEN", "PASSWORD", "SECRET", "KEY"]):
            env[k] = "***REDACTED***"
        else:
            env[k] = v

    return {
        "timestamp": datetime.now().isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "apptainer_version": _try(["apptainer", "--version"]),
        "nvidia_smi": _try(["nvidia-smi", "-L"]),
        "env": env,
    }
