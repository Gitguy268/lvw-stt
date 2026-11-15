"""Score a model per language on FLEURS test / FLEURS dev / CV22 test.

  python -m lvw_stt.evaluate --model phonon-base --tag baseline            # stored subsets
  python -m lvw_stt.evaluate --model qwen-ft --tag ft --full               # full FLEURS test, streamed from the Hub
  python -m lvw_stt.evaluate --gguf x.gguf [--mmproj m.gguf] --tag q4km    # GGUF via transcribe.cpp / llama-server

Writes results/<tag>-<model>-<set>.json (per-language error, average, RTF) and a .jsonl of hypotheses.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import socket
import subprocess
import tempfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from lvw_stt import langs
from lvw_stt.data import DATA, SR, decode, fleurs_rows, read_jsonl
from lvw_stt.disk import ROOT
from lvw_stt.paths import EN_NL_SETS, LLAMA_BIN, MODELS, PROFILE, PROFILE_LANGS, RESULTS, TRANSCRIBE_CLI
from lvw_stt.textnorm import score


def device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------- backends: list[np.ndarray] -> list[str]

class ParakeetHF:
    def __init__(self, path: Path, dtype: str = "float32", batch: int = 8):
        import torch
        from transformers import AutoProcessor
        from lvw_stt.parakeet.expand_phonon import load_parakeet
        self.torch = torch
        self.dev = device()
        self.model = load_parakeet(path, dtype=getattr(torch, dtype)).to(self.dev).eval()
        self.proc = AutoProcessor.from_pretrained(path)
        self.batch = batch

    def __call__(self, waves, codes=None):
        out = []
        for i in range(0, len(waves), self.batch):
            inp = self.proc(waves[i:i + self.batch], sampling_rate=SR, return_tensors="pt", padding=True)
            with self.torch.no_grad():
                seq = self.model.generate(input_features=inp["input_features"].to(self.dev, self.model.dtype),
                                          attention_mask=inp["attention_mask"].to(self.dev))
            seq = getattr(seq, "sequences", seq)
            out += [t.strip() for t in self.proc.batch_decode(seq, skip_special_tokens=True)]
        return out


class QwenHF:
    def __init__(self, path: Path, batch: int = 8):
        import torch
        from qwen_asr import Qwen3ASRModel
        self.m = Qwen3ASRModel.from_pretrained(str(path), dtype=torch.bfloat16, device_map=device(),
                                               max_inference_batch_size=batch, max_new_tokens=384)

    def __call__(self, waves, codes=None):
        # language=None, like Qwen's own benchmark runs (language ID is part of the task)
        res = self.m.transcribe(audio=[(w, SR) for w in waves], language=None)
        return [r.text.strip() for r in res]


class TranscribeCpp:
    """Parakeet GGUF through transcribe.cpp (Metal)."""
    def __init__(self, gguf: Path, batch: int = 8):
        self.gguf, self.batch = gguf, batch

    def __call__(self, waves, codes=None):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as td:
            files = []
            for i, w in enumerate(waves):
                p = Path(td) / f"{i:06d}.wav"
                sf.write(p, w, SR, subtype="PCM_16")
                files.append(str(p))
            lst = Path(td) / "list.txt"
            lst.write_text("\n".join(files) + "\n")
            r = subprocess.run([str(TRANSCRIBE_CLI), "-m", str(self.gguf), "--batch", str(lst), "--batch-jsonl",
                                "--batch-size", str(self.batch), "-q"], capture_output=True, text=True, check=True)
            rows = [json.loads(l) for l in r.stdout.splitlines() if l.strip().startswith("{")]
            by = {row["file"]: row.get("text", "") for row in rows if row.get("type") != "batch_header"}
            return [by.get(f, "").strip() for f in files]


class LlamaServer:
    """Qwen3-ASR GGUF (+ mmproj) through llama-server, loaded once."""
    def __init__(self, gguf: Path, mmproj: Path):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); self.port = s.getsockname()[1]; s.close()
        self.proc = subprocess.Popen([str(LLAMA_BIN / "llama-server"), "-m", str(gguf), "--mmproj", str(mmproj),
                                      "--port", str(self.port), "--host", "127.0.0.1", "-ngl", "99", "-c", "4096",
                                      "--temp", "0"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        else:
            self.close()
            raise RuntimeError("llama-server did not start")

    def one(self, w):
        b = io.BytesIO(); sf.write(b, w, SR, format="WAV", subtype="PCM_16")
        body = {"messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(b.getvalue()).decode(), "format": "wav"}}]}],
            "temperature": 0, "max_tokens": 384}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        txt = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]
        return parse_qwen(txt)

    def __call__(self, waves, codes=None):
        return [self.one(w) for w in waves]

    def close(self):
        self.proc.terminate()


def parse_qwen(raw: str) -> str:
    s = (raw or "").strip()
    if "<asr_text>" in s:
        head, s = s.split("<asr_text>", 1)
        if head.strip().lower() == "language none":
            return ""
    return s.strip()


def make_backend(a):
    if a.gguf:
        g = Path(a.gguf)
        return (LlamaServer(g, Path(a.mmproj)) if a.mmproj else TranscribeCpp(g)), ("qwen" if a.mmproj else "parakeet")
    kind, path = MODELS[a.model]
    if a.path:
        path = Path(a.path)
    if kind == "parakeet":
        return ParakeetHF(path, dtype=a.dtype), kind
    return QwenHF(path), kind


# ---------------------------------------------------------------- eval sets

def load_set(name: str, codes: list[str], full: bool, limit: int | None):
    """Yields (lang, id, wave, ref)."""
    if full and name == "fleurs-test":
        for code in codes:
            n = 0
            for rid, raw, text in fleurs_rows(code, "test", seed=0):
                if limit and n >= limit:
                    break
                yield code, f"fleurs-{code}-{rid}", decode(raw), text
                n += 1
        return
    rows = read_jsonl(DATA / "eval" / f"{name}.jsonl")
    if not rows:
        raise SystemExit(f"no {name} manifest; run: bash run.sh prepare-eval")
    per = defaultdict(int)
    for r in rows:
        if r["lang"] not in codes or (limit and per[r["lang"]] >= limit):
            continue
        per[r["lang"]] += 1
        w, _ = sf.read(ROOT / r["path"], dtype="float32")
        yield r["lang"], r["id"], w, r["text"]


def run(backend, kind: str, sets: list[str], codes: list[str], tag: str, label: str, full: bool,
        limit: int | None, chunk: int = 64) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in sets:
        refs, hyps = defaultdict(list), defaultdict(list)
        audio_s, wall = 0.0, 0.0
        out_jsonl = RESULTS / f"{tag}-{label}-{name}.jsonl"
        buf = []

        def flush():
            nonlocal audio_s, wall
            if not buf:
                return
            t0 = time.perf_counter()
            hs = backend([b[2] for b in buf], [b[0] for b in buf])
            wall += time.perf_counter() - t0
            with open(out_jsonl, "a", encoding="utf-8") as f:
                for (code, uid, w, ref), h in zip(buf, hs):
                    refs[code].append(ref); hyps[code].append(h)
                    audio_s += len(w) / SR
                    f.write(json.dumps({"id": uid, "lang": code, "ref": ref, "hyp": h}, ensure_ascii=False) + "\n")
            buf.clear()

        out_jsonl.unlink(missing_ok=True)
        set_codes = [c for c in codes if not name.startswith("cv22-test") or langs.cv(c)]
        for item in load_set(name, set_codes, full, limit):
            buf.append(item)
            if len(buf) >= chunk:
                flush()
        flush()
        per = {c: score(refs[c], hyps[c], c) for c in refs}
        vals = [v["err"] for v in per.values() if not math.isnan(v["err"])]
        res = {"model": label, "set": name + ("-full" if full else ""), "tag": tag, "kind": kind,
               "avg": round(sum(vals) / max(len(vals), 1), 3), "n_langs": len(vals),
               "rtf": round(audio_s / max(wall, 1e-9), 2), "audio_h": round(audio_s / 3600, 2),
               "per_lang": per}
        (RESULTS / f"{tag}-{label}-{name}{'-full' if full else ''}.json").write_text(json.dumps(res, indent=1))
        summary[name] = res
        print(f"{label:16s} {name:12s} avg {res['avg']:6.2f}  over {len(vals)} langs  RTFx {res['rtf']}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), help="named model (see paths.MODELS)")
    ap.add_argument("--path", help="override the model directory (e.g. a checkpoint)")
    ap.add_argument("--gguf"); ap.add_argument("--mmproj")
    ap.add_argument("--label", help="name in result files (default: model name or gguf stem)")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--sets", nargs="+",
                    default=EN_NL_SETS if PROFILE == "en-nl" else ["fleurs-test", "cv22-test"])
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--full", action="store_true", help="full FLEURS test sets streamed from the Hub")
    ap.add_argument("--limit", type=int, help="max utterances per language")
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    if not (a.model or a.gguf):
        ap.error("--model or --gguf")
    backend, kind = make_backend(a)
    label = a.label or a.model or Path(a.gguf).stem
    codes = a.langs or PROFILE_LANGS or (langs.PHONON_LANGS if kind == "parakeet" else langs.QWEN_LANGS)
    try:
        run(backend, kind, a.sets, codes, a.tag, label, a.full, a.limit)
    finally:
        if hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()
