"""Download raw connectome data and the optional flybody model.

Sources (verified 2026-10-03):
  MaleCNS v1.0   https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
                 (public GCS bucket gs://flyem-male-cns, CC BY 4.0)
  FlyWire v783   Completeness_783.csv + Connectivity_783.parquet from github.com/philshiu/Drosophila_brain_model
                 (MIT code; data CC BY-NC 4.0), annotations from github.com/flyconnectome/flywire_annotations
  flybody        github.com/TuragaLab/flybody (Apache-2.0), cloned into data/external/flybody
"""
from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

from .paths import DATA, RAW

MALECNS_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
MALECNS_FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
]
SHIU_RAW = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/"
FLYWIRE_ANN = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/"
               "Supplemental_file1_neuron_annotations.tsv")


def _fetch(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  exists: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def download_malecns(force: bool = False) -> None:
    for f in MALECNS_FILES:
        _fetch(MALECNS_BASE + f, RAW / "malecns_v1.0" / f, force)


def download_flywire(force: bool = False) -> None:
    d = RAW / "flywire_v783"
    _fetch(SHIU_RAW + "Completeness_783.csv", d / "Completeness_783.csv", force)
    _fetch(SHIU_RAW + "Connectivity_783.parquet", d / "Connectivity_783.parquet", force)
    _fetch(FLYWIRE_ANN, d / "Supplemental_file1_neuron_annotations.tsv", force)


def download_flybody() -> None:
    dest = DATA / "external" / "flybody"
    if dest.exists():
        print("  exists: flybody")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/TuragaLab/flybody.git", str(dest)], check=True)
