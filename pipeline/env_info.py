"""env_info.py - hardware/software facts stored in every result file (reproducibility)."""
import datetime
import os
import platform
import subprocess
import sys


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def collect_env():
    import psutil
    try:
        java = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()[0]
    except Exception:
        java = None
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    return {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "os": platform.platform(), "python": sys.version.split()[0],
        "cpu_model": _cpu_model(), "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False), "cores_usable": affinity,
        "ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 1),
        "java": java, "colab": "google.colab" in sys.modules or "COLAB_GPU" in os.environ,
        "packages": {m: _ver(m) for m in ["autosklearn", "h2o", "sklearn", "numpy", "pandas",
                                          "scipy", "psutil", "joblib"]},
    }
