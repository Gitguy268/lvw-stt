"""Stream open speech data from the Hub into small local shards under a disk budget.

  python -m lvw_stt.data eval     FLEURS test/dev + CV22 test subsets (FLAC, for scoring / checkpoint selection)
  python -m lvw_stt.data prepare  training audio: FLEURS train, CV22 train, VoxPopuli train (Opus or FLAC)
  python -m lvw_stt.data eval-en-nl   en-nl profile: ESB English test subsets, Dutch/English test sets, en+nl dev set

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
from lvw_stt.paths import EN_NL_DEV, ESB_SETS, PROFILE, TRAIN_DIR

DATA = ROOT / "data"
SR = 16000
MIN_S, MAX_S = 1.0, 20.0
FLEURS = "datasets/google/fleurs"
CV = "datasets/fsicoli/common_voice_22_0"
VOX = "datasets/facebook/voxpopuli"
ESB = "datasets/hf-audio/esb-datasets-test-only-sorted"
# set name -> (ESB config folder, split file prefix). These are the Open ASR Leaderboard's English test sets.
ESB_CFG = {"esb-ls-clean": ("librispeech", "test.clean"), "esb-ls-other": ("librispeech", "test.other"),
           "esb-ami": ("ami", "test"), "esb-earnings22": ("earnings22", "test"), "esb-gigaspeech": ("gigaspeech", "test"),
           "esb-spgispeech": ("spgispeech", "test"), "esb-voxpopuli": ("voxpopuli", "test")}
assert list(ESB_CFG) == ESB_SETS
ESB_SKIP = "ignore_time_segment_in_scoring"
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


def esb_rows(name: str, seed: int, per_group: int = 16):
    """One ESB English test set. The files are sorted by length, so reading the first rows would give a length-biased
    sample: row groups are visited in a seeded random order across all shards, and a few random rows are taken from
    each (cluster sampling; per_group x groups ~= the clip count asked for)."""
    import pyarrow.parquet as pq
    cfg, prefix = ESB_CFG[name]
    paths = sorted(fs().glob(f"{ESB}/{cfg}/{prefix}-*.parquet"))
    if not paths:
        raise RuntimeError(f"{ESB}/{cfg}/{prefix}-*.parquet: no files")
    pairs = []
    for p in paths:
        with fs().open(p, "rb", block_size=1 << 20) as f:
            pairs += [(p, g) for g in range(pq.ParquetFile(f).num_row_groups)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    for p, g in pairs:
        with fs().open(p, "rb", block_size=4 << 20) as f:
            t = pq.ParquetFile(f).read_row_group(g, columns=["audio", "text", "id"]).to_pylist()
        rng.shuffle(t)
        taken = 0
        for r in t:
            if taken >= per_group:
                break
            text = r.get("text") or ""
            a = r["audio"]
            raw = a.get("bytes") if isinstance(a, dict) else None
            if not raw or not text.strip() or ESB_SKIP in text:
                continue
            taken += 1
            yield str(r["id"]).replace("/", "_"), raw, text


def _esb_source(name: str):
    def rows(code: str, split: str, seed: int):
        assert code == "en" and split == "test", (name, code, split)
        yield from esb_rows(name, seed)
    return rows


SOURCES = {"fleurs": fleurs_rows, "cv22": cv_rows, "voxpopuli": voxpopuli_rows,
           **{n: _esb_source(n) for n in ESB_CFG}}


# ---------------------------------------------------------------- collection

def collect(out: Path, code: str, source: str, split: str, *, max_utts: int | None = None,
            max_hours: float | None = None, codec: str = "flac", seed: int = 0, seen: set | None = None,
            min_s: float = MIN_S, max_s: float = MAX_S):
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
        if not (min_s <= dur <= max_s):
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
    if a.mix:                                   # explicit language shares (en-nl profile), no baseline weighting
        w = parse_mix(a.mix)
        codes = list(w)
    else:
        codes = a.langs or langs.ALL_LANGS
        w = weights(codes, a.tau)
    man = TRAIN_DIR / "manifest.jsonl"
    have = read_jsonl(man)
    seen = {r["id"] for r in have}
    got = defaultdict(float)
    for r in have:
        got[(r["lang"], r["source"])] += r["dur"] / 3600
    split_share = parse_mix(a.shares) if a.shares else {"fleurs": 0.45, "cv22": 0.35, "voxpopuli": 0.20}
    assert set(split_share) <= {"fleurs", "cv22", "voxpopuli"}, split_share
    print(f"language shares: {w}; source shares: {split_share}")
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
            rows = collect(TRAIN_DIR, code, source, "train", max_hours=want, codec=codec, seed=a.seed, seen=seen)
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
    from lvw_stt.paths import RESULTS
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "train-data.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def parse_mix(s: str) -> dict[str, float]:
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    tot = sum(out.values())
    if not out or tot <= 0:
        raise SystemExit(f"bad mix {s!r}")
    return {k: v / tot for k, v in out.items()}


def _fill(name: str, code: str, source: str, split: str, n: int, *, min_s=MIN_S, max_s=MAX_S,
          copy_from: str | None = None) -> None:
    """Top manifest data/eval/<name>.jsonl up to n clips of `code`. copy_from re-uses clips already on disk from the
    EU eval set (same files, no download) and then adds new, different clips."""
    out = DATA / "eval"
    man = out / f"{name}.jsonl"
    have = [r for r in read_jsonl(man) if r["lang"] == code]
    if copy_from and not have:
        old = [r for r in read_jsonl(out / f"{copy_from}.jsonl") if r["lang"] == code][:n]
        append_jsonl(man, old)
        have = old
    if len(have) >= n:
        print(f"{name:20s} {code}: {len(have)} clips (done)")
        return
    seen = {r["id"] for r in read_jsonl(man)}
    if copy_from:
        seen |= {r["id"] for r in read_jsonl(out / f"{copy_from}.jsonl")}
    rows = collect(out / name, code, source, split, max_utts=n - len(have), codec="flac", seed=1, seen=seen,
                   min_s=min_s, max_s=max_s)
    append_jsonl(man, rows)
    print(f"{name:20s} {code}: +{len(rows)} -> {len(have) + len(rows)} clips  ({free_gb():.1f} GB free)")


def stage_eval_en_nl(a) -> None:
    """en-nl scoring + dev sets. Test splits only for scoring; dev for checkpoint selection; never train splits."""
    if PROFILE != "en-nl":
        raise SystemExit("run with LVW_PROFILE=en-nl")
    n = a.per_set
    guard(1.0, what="en-nl eval sets")
    for name in ESB_SETS:                                       # the 7 English sets on Phonon-2.5-English's card
        _fill(name, "en", name, "test", n, min_s=0.0, max_s=60.0)
    for code in ("en", "nl"):                                   # FLEURS test: all sentences (one recording each)
        _fill("fleurs-test-en-nl", code, "fleurs", "test", n, min_s=0.0, max_s=60.0, copy_from="fleurs-test")
        _fill("cv22-test-en-nl", code, "cv22", "test", n, copy_from="cv22-test")
    _fill("voxpopuli-test-nl", "nl", "voxpopuli", "test", n, min_s=0.0, max_s=60.0)
    # dev: FLEURS dev (already on disk) + VoxPopuli validation + CV22 dev, per language
    dev = EN_NL_DEV
    have = read_jsonl(dev)
    if not have:
        rows = [r for r in read_jsonl(DATA / "eval" / "fleurs-dev.jsonl") if r["lang"] in ("en", "nl")]
        append_jsonl(dev, rows)
        have = rows
    count = defaultdict(int)
    for r in have:
        count[(r["lang"], r["source"])] += 1
    seen = {r["id"] for r in have}
    for code in ("en", "nl"):
        for source in ("voxpopuli", "cv22"):
            want = 60 - count[(code, source)]
            if want > 0:
                rows = collect(DATA / "eval" / "dev-en-nl", code, source, "validation", max_utts=want, codec="flac",
                               seed=1, seen=seen)
                append_jsonl(dev, rows)
                print(f"dev-en-nl            {code} {source}: +{len(rows)}")
    report_eval_en_nl()


def report_eval_en_nl() -> None:
    from lvw_stt.paths import EN_NL_SETS
    for name in EN_NL_SETS + ["dev-en-nl"]:
        rows = read_jsonl(DATA / "eval" / f"{name}.jsonl")
        by = defaultdict(lambda: [0, 0.0])
        for r in rows:
            by[r["lang"]][0] += 1; by[r["lang"]][1] += r["dur"] / 3600
        print(f"  {name:20s} " + "  ".join(f"{c}: {k} clips {h:.2f} h" for c, (k, h) in sorted(by.items())))


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
    p.add_argument("--mix", help="explicit language shares, e.g. en=0.4,nl=0.6 (replaces baseline weighting)")
    p.add_argument("--shares", help="source shares, e.g. fleurs=0.15,cv22=0.45,voxpopuli=0.40")
    en = sub.add_parser("eval-en-nl")
    en.add_argument("--per-set", type=int, default=400)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "eval":
        stage_eval(a)
    elif a.cmd == "prepare":
        stage_prepare(a)
    elif a.cmd == "eval-en-nl":
        stage_eval_en_nl(a)
    else:
        report(TRAIN_DIR / "manifest.jsonl")


if __name__ == "__main__":
    main()
