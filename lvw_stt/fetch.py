"""Download the pinned source models into runs/src (no shared HF cache copies).

  python -m lvw_stt.fetch --phonon-rev R --qwen-rev R --parakeet-rev R    # eu profile (setup)
  python -m lvw_stt.fetch --phonon-en-rev R                               # en-nl profile: Phonon-2.5-English only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

from lvw_stt.disk import ROOT, guard
from lvw_stt.paths import PHONON_SOURCES, SRC_PARAKEET_CFG, SRC_PHONON, SRC_QWEN

SRC = ROOT / "runs" / "src"
PHONON = SRC_PHONON
QWEN = SRC_QWEN
PARAKEET_CFG = SRC_PARAKEET_CFG


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def unpacked(src: Path) -> Path:
    """The .tar.zst has its files at the top level (incl. its own config.json), so it unpacks to <src>/unpacked."""
    return src / "unpacked"


UNPACKED = unpacked(PHONON)


def phonon_container(src: Path = PHONON) -> Path:
    cfg = json.loads((src / "config.json").read_text())["artifact"]
    return unpacked(src) / cfg["container"]


def fetch_phonon(key: str, rev: str) -> Path:
    """Download one Fermion Phonon repo at a pinned revision, verify bundle + container checksums, unpack."""
    s = PHONON_SOURCES[key]
    if rev != s["rev"]:
        raise SystemExit(f"{s['repo']}: asked for revision {rev}, pinned is {s['rev']}")
    src: Path = s["src"]
    guard(0.6, what=s["repo"])
    snapshot_download(s["repo"], revision=rev, local_dir=src,
                      ignore_patterns=["*.fsq", "*.jpg"])  # the .fsq needs Fermion's CLI; the tar.zst is the same container
    art = json.loads((src / "config.json").read_text())["artifact"]
    if s["tar_sha256"] and art["sha256"] != s["tar_sha256"]:
        raise SystemExit(f"{s['repo']}: config.json lists bundle sha {art['sha256']}, pinned is {s['tar_sha256']}")
    if s["container_sha256"] and art["container_sha256"] != s["container_sha256"]:
        raise SystemExit(f"{s['repo']}: config.json lists container sha {art['container_sha256']}, "
                         f"pinned is {s['container_sha256']}")
    bundle = src / art["filename"]
    if sha256(bundle) != art["sha256"]:
        raise SystemExit(f"{s['repo']}: bundle checksum mismatch ({bundle})")
    cont = phonon_container(src)
    if not cont.exists():
        import zstandard
        with open(bundle, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr, \
                tarfile.open(fileobj=zr, mode="r|") as tf:
            tf.extractall(unpacked(src), filter="data")
    if sha256(cont) != art["container_sha256"]:
        raise SystemExit(f"{s['repo']}: container checksum mismatch ({cont})")
    print(f"{s['repo']} ok: {cont}")
    return cont


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phonon-rev")
    ap.add_argument("--phonon-en-rev")
    ap.add_argument("--qwen-rev")
    ap.add_argument("--parakeet-rev")
    a = ap.parse_args()
    if not any((a.phonon_rev, a.phonon_en_rev, a.qwen_rev, a.parakeet_rev)):
        ap.error("nothing to fetch")
    SRC.mkdir(parents=True, exist_ok=True)

    if a.phonon_rev:
        fetch_phonon("multi", a.phonon_rev)
    if a.phonon_en_rev:
        fetch_phonon("en", a.phonon_en_rev)
        if not (PARAKEET_CFG / "config.json").exists():
            raise SystemExit("parakeet-tdt-0.6b-v3 config missing: run `bash run.sh setup` first")

    if a.qwen_rev:
        guard(2.0, what="Qwen3-ASR-0.6B")
        snapshot_download("Qwen/Qwen3-ASR-0.6B", revision=a.qwen_rev, local_dir=QWEN)
        print("qwen ok:", QWEN)

    if a.parakeet_rev:
        snapshot_download("nvidia/parakeet-tdt-0.6b-v3", revision=a.parakeet_rev, local_dir=PARAKEET_CFG,
                          allow_patterns=["config.json", "generation_config.json", "processor_config.json",
                                          "tokenizer.json", "tokenizer_config.json", "README.md"])
        print("parakeet-v3 config ok:", PARAKEET_CFG)


if __name__ == "__main__":
    main()
