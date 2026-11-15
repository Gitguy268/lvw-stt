"""Stream open speech data from the Hub into small local shards under a disk budget.

  python -m lvw_stt.data eval     FLEURS test/dev + CV22 test subsets (FLAC, for scoring / checkpoint selection)
  python -m lvw_stt.data prepare  training audio: FLEURS train, CV22 train, VoxPopuli train (Opus or FLAC)

Nothing here ever reads a test split into the training manifest. Audio is read with HTTP range requests
(parquet row groups / tar streams), so only the kept clips touch the disk.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from lvw_stt import langs
from lvw_stt.disk import ROOT, free_gb, guard

DATA = ROOT / "data"
SR = 16000
MIN_S, MAX_S = 1.0, 20.0
FLEURS = "datasets/google/fleurs"
CV = "datasets/fsicoli/common_voice_22_0"
VOX = "datasets/facebook/voxpopuli"
TEXT_COLS = ("raw_transcription", "raw_text", "sentence", "text")

_fs = None


def fs():
    global _fs
    if _fs is None:
        from huggingface_hub import HfFileSystem
        _fs = HfFileSystem()
    return _fs


# ---------------------------------------------------------------- audio

def decode(raw: bytes) -> np.ndarray:
    """bytes (wav/flac/mp3/ogg) -> float32 mono 16 kHz."""
    try:
        w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        w = w.mean(axis=1)
    except Exception:
        p = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                           input=raw, capture_output=True, check=True)
        return np.frombuffer(p.stdout, dtype=np.float32).copy()
    if sr != SR:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    return w.astype(np.float32)


def opus_ok() -> bool:
    try:
        b = io.BytesIO()
        sf.write(b, np.zeros(SR, dtype=np.float32), SR, format="OGG", subtype="OPUS")
        b.seek(0)
        sf.read(b)
        return True
    except Exception:
        return False


def write_audio(path: Path, w: np.ndarray, codec: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if codec == "opus":
        path = path.with_suffix(".ogg")
        sf.write(path, w, SR, format="OGG", subtype="OPUS", compression_level=0.6)
    else:
        path = path.with_suffix(".flac")
        sf.write(path, w, SR, format="FLAC", subtype="PCM_16")
    return path


def clean(text: str) -> str:
    return " ".join((text or "").replace(" ", " ").split())


# ---------------------------------------------------------------- sources (yield (id, raw_bytes, text))

def _parquet_rows(paths: list[str], seed: int):
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    for p in paths:
        with fs().open(p, "rb", block_size=4 << 20) as f:
            pf = pq.ParquetFile(f)
            names = pf.schema_arrow.names
            tcol = next((c for c in TEXT_COLS if c in names), None)
            if tcol is None or "audio" not in names:
                raise RuntimeError(f"{p}: no text/audio column in {names}")
            extra = [c for c in ("id", "audio_id", "is_gold_transcript") if c in names]
            groups = list(range(pf.num_row_groups))
            rng.shuffle(groups)
            for g in groups:
                t = pf.read_row_group(g, columns=["audio", tcol, *extra]).to_pylist()
                rng.shuffle(t)
                for r in t:
                    if r.get("is_gold_transcript") is False:
                        continue
                    a = r["audio"]
                    raw = a.get("bytes") if isinstance(a, dict) else None
                    if not raw:
                        continue
                    rid = str(r.get("id") or r.get("audio_id") or (a.get("path") or "")).replace("/", "_")
                    yield rid, raw, r[tcol]


def fleurs_rows(code: str, split: str, seed: int):
    cfg = langs.fleurs(code)
    paths = fs().glob(f"{FLEURS}/parquet-data/{cfg}/{split}-*.parquet")
    if not paths:
        print(f"  fleurs {cfg}/{split}: no parquet", file=sys.stderr)
        return
    yield from _parquet_rows(paths, seed)


def voxpopuli_rows(code: str, split: str, seed: int):
    cfg = langs.voxpopuli(code)
    if not cfg:
        return
    paths = fs().glob(f"{VOX}/{cfg}/{split}-*.parquet")
    yield from _parquet_rows(paths, seed)


def cv_rows(code: str, split: str, seed: int):
    folder = langs.cv(code)
    if not folder or not fs().exists(f"{CV}/transcript/{folder}/{split}.tsv"):
        if folder:
            print(f"  cv22 {folder}/{split}: missing", file=sys.stderr)
        return
    with fs().open(f"{CV}/transcript/{folder}/{split}.tsv", "r", encoding="utf-8") as f:
        rows = {r["path"]: r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    tars = sorted(fs().glob(f"{CV}/audio/{folder}/{split}/*.tar"))
    random.Random(seed).shuffle(tars)
    for t in tars:
        with fs().open(t, "rb", block_size=8 << 20) as fh, tarfile.open(fileobj=fh, mode="r|") as tf:
            for m in tf:
                name = Path(m.name).name
                r = rows.get(name)
                if r is None or not m.isfile() or int(r.get("down_votes") or 0) > int(r.get("up_votes") or 0):
                    continue
                yield name.rsplit(".", 1)[0], tf.extractfile(m).read(), r["sentence"]


SOURCES = {"fleurs": fleurs_rows, "cv22": cv_rows, "voxpopuli": voxpopuli_rows}


# ---------------------------------------------------------------- collection

def collect(out: Path, code: str, source: str, split: str, *, max_utts: int | None = None,
            max_hours: float | None = None, codec: str = "flac", seed: int = 0, seen: set | None = None):
    """Write up to max_utts / max_hours clips; returns manifest rows. Resumable via `seen`."""
    rows, hours = [], 0.0
    seen = seen or set()
    gen = SOURCES[source](code, "dev" if (source == "cv22" and split == "validation") else split, seed)
    for i, (rid, raw, text) in enumerate(gen):
        if (max_utts and len(rows) >= max_utts) or (max_hours and hours >= max_hours):
            break
        text = clean(text)
        rid = re.sub(r"[^\w.-]", "_", rid)
        uid = f"{source}-{code}-{rid}"
        if not text or uid in seen:
            continue
        try:
            w = decode(raw)
        except Exception as e:  # corrupt clip: skip, keep going
            print(f"  skip {uid}: {e}", file=sys.stderr)
            continue
        dur = len(w) / SR
        if not (MIN_S <= dur <= MAX_S):
            continue
        if len(rows) % 200 == 0:
            guard(0.3, what=f"{source}/{code}")
        p = write_audio(out / code / source / rid, w, codec)
        rows.append({"id": uid, "lang": code, "source": source, "split": split,
                     "path": str(p.relative_to(ROOT)), "text": text, "dur": round(dur, 3)})
        seen.add(uid)
        hours += dur / 3600
    return rows


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- stages

def stage_eval(a) -> None:
    """Scoring sets: FLEURS test (per-lang subset), FLEURS validation (checkpoint selection), CV22 test."""
    out = DATA / "eval"
    plan = [("fleurs-test", "fleurs", "test", a.per_lang), ("fleurs-dev", "fleurs", "validation", a.dev_per_lang),
            ("cv22-test", "cv22", "test", a.cv_per_lang)]
    for name, source, split, n in plan:
        man = out / f"{name}.jsonl"
        have = read_jsonl(man)
        done = defaultdict(int)
        for r in have:
            done[r["lang"]] += 1
        seen = {r["id"] for r in have}
        for code in (a.langs or langs.ALL_LANGS):
            if done[code] >= n:
                continue
            rows = collect(out / name, code, source, split, max_utts=n - done[code], codec="flac", seed=1, seen=seen)
            append_jsonl(man, rows)
            print(f"{name:12s} {code:4s} +{len(rows):4d}  ({free_gb():.1f} GB free)")


def weights(codes: list[str], tau: float) -> dict[str, float]:
    """Sampling weight per language from baseline error rates (higher error -> more data); uniform if no baseline."""
    err = {}
    for f in (ROOT / "results").glob("baseline-*-fleurs-test.json"):
        for code, v in json.loads(f.read_text())["per_lang"].items():
            err[code] = max(err.get(code, 0.0), v["err"])
    w = {c: (max(err.get(c, 10.0), 2.0)) ** tau for c in codes}
    s = sum(w.values())
    return {c: v / s for c, v in w.items()}


def stage_prepare(a) -> None:
    codec = a.codec
    if codec == "opus" and not opus_ok():
        print("soundfile cannot write Opus here; using FLAC")
        codec = "flac"
    mb_per_hour = 15 if codec == "opus" else 65   # 32 kbps Opus vs 16-bit FLAC speech
    hours_total = a.gb * 1000 / mb_per_hour
    codes = a.langs or langs.ALL_LANGS
    w = weights(codes, a.tau)
    man = DATA / "train" / "manifest.jsonl"
    have = read_jsonl(man)
    seen = {r["id"] for r in have}
    got = defaultdict(float)
    for r in have:
        got[(r["lang"], r["source"])] += r["dur"] / 3600
    split_share = {"fleurs": 0.45, "cv22": 0.35, "voxpopuli": 0.20}
    print(f"budget {a.gb} GB ≈ {hours_total:.0f} h ({codec}); {len(codes)} languages")
    for code in codes:
        quota = hours_total * w[code]
        avail = [s for s in split_share if (s != "voxpopuli" or langs.voxpopuli(code)) and (s != "cv22" or langs.cv(code))]
        norm = sum(split_share[s] for s in avail)
        carry = 0.0
        for source in avail:
            want = quota * split_share[source] / norm + carry - got[(code, source)]
            if want <= 0.01:
                carry = 0.0
                continue
            rows = collect(DATA / "train", code, source, "train", max_hours=want, codec=codec, seed=a.seed, seen=seen)
            h = sum(r["dur"] for r in rows) / 3600
            carry = max(0.0, want - h)          # FLEURS caps at ~10 h/lang; spill to the next source
            append_jsonl(man, rows)
            print(f"{code:4s} {source:9s} +{h:5.2f} h / want {want:5.2f}  ({free_gb():.1f} GB free)")
    report(man)


def report(man: Path) -> None:
    rows = read_jsonl(man)
    by = defaultdict(lambda: defaultdict(float))
    for r in rows:
        by[r["lang"]][r["source"]] += r["dur"] / 3600
    lines = ["| lang | fleurs h | cv22 h | voxpopuli h | total h |", "|---|--:|--:|--:|--:|"]
    for code in sorted(by):
        d = by[code]
        lines.append(f"| {code} | {d['fleurs']:.2f} | {d['cv22']:.2f} | {d['voxpopuli']:.2f} | {sum(d.values()):.2f} |")
    tot = sum(sum(d.values()) for d in by.values())
    lines.append(f"| **all** | | | | **{tot:.1f}** |")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "train-data.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--per-lang", type=int, default=200)
    e.add_argument("--dev-per-lang", type=int, default=40)
    e.add_argument("--cv-per-lang", type=int, default=150)
    e.add_argument("--langs", nargs="*")
    p = sub.add_parser("prepare")
    p.add_argument("--gb", type=float, default=5.0, help="audio budget on disk")
    p.add_argument("--codec", choices=["opus", "flac"], default="opus")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--langs", nargs="*")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "eval":
        stage_eval(a)
    elif a.cmd == "prepare":
        stage_prepare(a)
    else:
        report(DATA / "train" / "manifest.jsonl")


if __name__ == "__main__":
    main()
