"""
Reproducibility helpers: unique run folders, run metadata and the trial log.

Each run writes to results/<data-source>/runs/<UTC time>_<short id>/ and never
touches earlier outputs. run.json records the configuration, the Git commit
and whether the working tree had uncommitted changes, package versions,
hashes of the source files and the data conventions used.
"""

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost", "pyarrow", "matplotlib", "torch",
            "timestral", "ib_async"]


def new_run_dir(results_dir, source, tag=""):
    base = Path(results_dir)
    base = base if base.is_absolute() else PROJECT_DIR / base
    name = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}" + (f"_{tag}" if tag else "")
    d = base / source / "runs" / name
    d.mkdir(parents=True, exist_ok=False)
    return d


def git_state():
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=PROJECT_DIR, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    commit = run("rev-parse", "HEAD")
    return {"commit": commit or "unknown", "dirty": bool(run("status", "--porcelain")) if commit else None}


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def source_hashes():
    files = sorted(PROJECT_DIR.glob("*.py")) + sorted((PROJECT_DIR / "tools").glob("*.py"))
    return {str(f.relative_to(PROJECT_DIR)): hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            for f in files}


def config_snapshot(cfg_module):
    return {k: v for k, v in vars(cfg_module).items()
            if k.isupper() and isinstance(v, (int, float, str, bool, list, tuple, dict, type(None)))}


def write_run_json(run_dir, payload):
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "git": git_state(),
               "versions": versions(), "source_sha256_16": source_hashes(), **payload}
    (Path(run_dir) / "run.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload


def append_trial(results_dir, source, row):
    """One line per run and model: keeps a record of every trial made.

    Runs whose test period was already inspected are marked exploratory.
    """
    import pandas as pd
    base = Path(results_dir)
    base = base if base.is_absolute() else PROJECT_DIR / base
    f = base / source / "trial_log.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)
