"""Central location of all project paths (overridable via environment variables)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("FLYLAB_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("FLYLAB_DATA", ROOT / "data"))
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RUNS = Path(os.environ.get("FLYLAB_RUNS", ROOT / "runs"))
CONFIGS = ROOT / "configs"
ASSETS = Path(__file__).resolve().parent / "body" / "assets"
