"""Download the pinned source models into runs/src (no shared HF cache copies)."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

from lvw_stt.disk import ROOT, guard

SRC = ROOT / "runs" / "src"
PHONON = SRC / "phonon-2.5"
QWEN = SRC / "qwen3-asr-0.6b"
PARAKEET_CFG = SRC / "parakeet-tdt-0.6b-v3"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


UNPACKED = PHONON / "unpacked"     # the .tar.zst has its files at the top level (incl. its own config.json)


def phonon_container() -> Path:
    cfg = json.loads((PHONON / "config.json").read_text())["artifact"]
    return UNPACKED / cfg["container"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phonon-rev", required=True)
    ap.add_argument("--qwen-rev", required=True)
    ap.add_argument("--parakeet-rev", required=True)
    a = ap.parse_args()
    SRC.mkdir(parents=True, exist_ok=True)

    guard(0.6, what="Phonon-2.5")
    snapshot_download("FermionResearch/Phonon-2.5", revision=a.phonon_rev, local_dir=PHONON,
                      ignore_patterns=["*.fsq"])  # the .fsq needs Fermion's CLI; the tar.zst is the same container
    art = json.loads((PHONON / "config.json").read_text())["artifact"]
    bundle = PHONON / art["filename"]
    assert sha256(bundle) == art["sha256"], "Phonon bundle checksum mismatch"
    cont = phonon_container()
    if not cont.exists():
        import zstandard
        with open(bundle, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr, \
                tarfile.open(fileobj=zr, mode="r|") as tf:
            tf.extractall(UNPACKED, filter="data")
    assert sha256(cont) == art["container_sha256"], "Phonon container checksum mismatch"
    print("phonon ok:", cont)

    guard(2.0, what="Qwen3-ASR-0.6B")
    snapshot_download("Qwen/Qwen3-ASR-0.6B", revision=a.qwen_rev, local_dir=QWEN)
    print("qwen ok:", QWEN)

    snapshot_download("nvidia/parakeet-tdt-0.6b-v3", revision=a.parakeet_rev, local_dir=PARAKEET_CFG,
                      allow_patterns=["config.json", "generation_config.json", "processor_config.json",
                                      "tokenizer.json", "tokenizer_config.json", "README.md"])
    print("parakeet-v3 config ok:", PARAKEET_CFG)


if __name__ == "__main__":
    main()
