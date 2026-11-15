"""GGUF ladders, one file at a time: convert -> parity -> quantize -> score -> (upload) -> delete.

  python -m lvw_stt.gguf.ladder --family eu                 # Parakeet side via transcribe.cpp
  python -m lvw_stt.gguf.ladder --family qwen               # Qwen side via llama.cpp (decoder ladder + mmproj)
  python -m lvw_stt.gguf.ladder --family eu --quant Q4_K_M  # one quant
  python -m lvw_stt.gguf.ladder --family qwen --upload      # also upload each file to the (private) GGUF repo

Without --upload every quant is deleted after scoring (the disk cannot hold both ladders); re-quantizing for the
upload takes seconds. Results: results/gguf-<family>.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from lvw_stt.disk import ROOT, guard
from lvw_stt.paths import EU_NAME, HF_USER, LLAMA_BIN, QW_NAME, RELEASE, RESULTS, RUNS, TP, TRANSCRIBE_QUANT

PY_PK = ROOT / "envs" / "parakeet" / "bin" / "python"
PY_QW = ROOT / "envs" / "qwen" / "bin" / "python"
WORK = RUNS / "gguf"

EU_QUANTS = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_1", "Q5_0", "Q4_K_M", "Q4_1", "Q4_0"]   # transcribe-quantize presets
QW_QUANTS = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_0", "IQ4_XS", "IQ4_NL",
             "Q3_K_L", "Q3_K_M", "Q3_K_S", "IQ3_M", "Q2_K", "IQ2_M"]
NEEDS_IMATRIX = {"IQ4_XS", "IQ4_NL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "IQ3_M", "Q2_K", "IQ2_M"}
MMPROJ_TYPES = ["f16", "bf16", "q8_0"]
LIMIT = 50       # FLEURS-test utterances per language for every quant
PARITY_MIN = {"eu": 0.98, "qwen": 0.90}


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def evaluate(py: Path, args: list[str], label: str, tag: str = "gguf") -> dict:
    sh([py, "-m", "lvw_stt.evaluate", *args, "--tag", tag, "--label", label, "--sets", "fleurs-test", "--limit", LIMIT],
       cwd=ROOT)
    return json.loads((RESULTS / f"{tag}-{label}-fleurs-test.json").read_text())


def parity(label_a: str, label_b: str, tag_a: str, tag_b: str) -> float:
    def hyps(tag, label):
        rows = [json.loads(l) for l in (RESULTS / f"{tag}-{label}-fleurs-test.jsonl").read_text().splitlines()]
        return {r["id"]: r["hyp"].strip() for r in rows}
    a, b = hyps(tag_a, label_a), hyps(tag_b, label_b)
    ids = sorted(set(a) & set(b))
    return sum(a[i] == b[i] for i in ids) / max(1, len(ids))


class Uploader:
    def __init__(self, repo: str | None):
        self.repo = repo
        if repo:
            from huggingface_hub import HfApi
            self.api = HfApi()
            self.api.create_repo(repo, private=True, exist_ok=True)

    def __call__(self, f: Path):
        if self.repo:
            self.api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=self.repo)
            print(f"uploaded {f.name} -> {self.repo}")


def record(family: str, row: dict) -> None:
    p = RESULTS / f"gguf-{family}.json"
    rows = json.loads(p.read_text()) if p.exists() else []
    rows = [r for r in rows if r["file"] != row["file"]] + [row]
    p.write_text(json.dumps(rows, indent=1))


def ladder_eu(a, fast: bool) -> None:
    name = f"{EU_NAME}-Fast" if fast else EU_NAME
    src = RELEASE / name
    key = name.lower()
    d = WORK / key
    d.mkdir(parents=True, exist_ok=True)
    top = d / f"{key}-F32.gguf"                       # folder + file name select the converter profile
    if not top.exists():
        guard(2.6, what="F32 GGUF")
        sh([PY_PK, TP / "transcribe.cpp" / "scripts" / "convert-parakeet.py", src, top], cwd=TP / "transcribe.cpp")
    family = "eu-fast" if fast else "eu"
    hf_label = "parakeet-fast" if fast else "parakeet-ft"
    evaluate(PY_PK, ["--model", hf_label], f"{hf_label}-hf", tag="gguf-ref")
    pub = d / f"{name}-F32.gguf"
    shutil.move(top, pub) if not pub.exists() else None
    ref = evaluate(PY_PK, ["--gguf", pub], pub.stem)
    p = parity(f"{hf_label}-hf", pub.stem, "gguf-ref", "gguf")
    print(f"parity F32 GGUF vs HF: {100 * p:.1f}% identical transcripts")
    if p < PARITY_MIN["eu"]:
        sys.exit(f"parity {p:.3f} < {PARITY_MIN['eu']}: fix conversion before quantizing")
    up = Uploader(a.upload and f"{HF_USER}/{name}-GGUF")
    record(family, {"file": pub.name, "quant": "F32", "bytes": pub.stat().st_size, "avg": ref["avg"], "rtf": ref["rtf"],
                    "parity": round(p, 4), "rel_vs_top_pct": 0.0})
    up(pub)
    for q in (a.quant or EU_QUANTS):
        guard(1.5, what=q)
        f = d / f"{name}-{q}.gguf"
        sh([TRANSCRIBE_QUANT, pub, f, "--quant", q])
        r = evaluate(PY_PK, ["--gguf", f], f.stem)
        record(family, {"file": f.name, "quant": q, "bytes": f.stat().st_size, "avg": r["avg"], "rtf": r["rtf"],
                        "rel_vs_top_pct": round(100 * (r["avg"] - ref["avg"]) / ref["avg"], 2)})
        up(f)
        f.unlink()
    if not a.keep_top:
        pub.unlink()


def ladder_qwen(a, fast: bool) -> None:
    name = f"{QW_NAME}-Fast" if fast else QW_NAME
    src = RELEASE / name
    d = WORK / name.lower()
    d.mkdir(parents=True, exist_ok=True)
    conv = TP / "llama.cpp" / "convert_hf_to_gguf.py"
    top = d / f"{name}-BF16.gguf"
    if not top.exists():
        guard(1.6, what="BF16 GGUF")
        sh([PY_QW, conv, src, "--outfile", top, "--outtype", "bf16"])
    mm = {}
    for t in MMPROJ_TYPES:
        f = d / f"mmproj-{name}-{t.upper()}.gguf"
        if not f.exists():
            sh([PY_QW, conv, src, "--mmproj", "--outfile", f, "--outtype", t])
        mm[t] = f
    family = "qwen-fast" if fast else "qwen"
    hf_label = "qwen-fast" if fast else "qwen-ft"
    evaluate(PY_QW, ["--model", hf_label], f"{hf_label}-hf", tag="gguf-ref")
    ref = evaluate(PY_QW, ["--gguf", top, "--mmproj", mm["f16"]], top.stem)
    p = parity(f"{hf_label}-hf", top.stem, "gguf-ref", "gguf")
    print(f"parity BF16 GGUF vs HF bf16: {100 * p:.1f}% identical transcripts")
    if p < PARITY_MIN["qwen"]:
        sys.exit(f"parity {p:.3f} < {PARITY_MIN['qwen']}: fix conversion before quantizing")
    up = Uploader(a.upload and f"{HF_USER}/{name}-GGUF")
    record(family, {"file": top.name, "quant": "BF16", "bytes": top.stat().st_size, "avg": ref["avg"], "rtf": ref["rtf"],
                    "parity": round(p, 4), "rel_vs_top_pct": 0.0})
    up(top)
    for t, f in mm.items():
        if t != "f16":
            r = evaluate(PY_QW, ["--gguf", top, "--mmproj", f], f"{top.stem}+{f.stem}")
            record(family, {"file": f.name, "quant": f"mmproj-{t.upper()}", "bytes": f.stat().st_size, "avg": r["avg"],
                            "rtf": r["rtf"], "rel_vs_top_pct": round(100 * (r["avg"] - ref["avg"]) / ref["avg"], 2)})
        else:
            record(family, {"file": f.name, "quant": "mmproj-F16", "bytes": f.stat().st_size, "avg": ref["avg"],
                            "rtf": ref["rtf"], "rel_vs_top_pct": 0.0})
        up(f)
    imatrix = d / "imatrix.dat"
    quants = a.quant or QW_QUANTS
    if any(q in NEEDS_IMATRIX for q in quants) and not imatrix.exists():
        calib = d / "calib.txt"
        from lvw_stt.data import DATA, read_jsonl
        from lvw_stt.langs import QWEN_LANGS
        lines = [r["text"] for r in read_jsonl(DATA / "train" / "manifest.jsonl") if r["lang"] in QWEN_LANGS]
        calib.write_text("\n".join(lines[::max(1, len(lines) // 6000)]) + "\n")
        sh([LLAMA_BIN / "llama-imatrix", "-m", top, "-f", calib, "-o", imatrix, "--chunks", "200", "-ngl", "99"])
    for q in quants:
        guard(1.0, what=q)
        f = d / f"{name}-{q}.gguf"
        cmd = [LLAMA_BIN / "llama-quantize"]
        if q in NEEDS_IMATRIX:
            cmd += ["--imatrix", imatrix]
        sh(cmd + [top, f, q])
        r = evaluate(PY_QW, ["--gguf", f, "--mmproj", mm["f16"]], f.stem)
        record(family, {"file": f.name, "quant": q, "bytes": f.stat().st_size, "avg": r["avg"], "rtf": r["rtf"],
                        "imatrix": q in NEEDS_IMATRIX, "rel_vs_top_pct": round(100 * (r["avg"] - ref["avg"]) / ref["avg"], 2)})
        up(f)
        f.unlink()
    if not a.keep_top:
        top.unlink()
        for f in mm.values():
            f.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["eu", "qwen", "eu-fast", "qwen-fast", "all"], default="all")
    ap.add_argument("--quant", nargs="*")
    ap.add_argument("--upload", action="store_true", help="upload every file to <user>/<model>-GGUF (private)")
    ap.add_argument("--keep-top", action="store_true")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    fams = ["eu", "qwen", "eu-fast", "qwen-fast"] if a.family == "all" else [a.family]
    for fam in fams:
        fast = fam.endswith("-fast")
        src = RELEASE / ((EU_NAME if fam.startswith("eu") else QW_NAME) + ("-Fast" if fast else ""))
        if not (src / "model.safetensors").exists() and not any(src.glob("model*.safetensors")):
            print(f"{fam}: no release at {src}; skipped"); continue
        (ladder_eu if fam.startswith("eu") else ladder_qwen)(a, fast)


if __name__ == "__main__":
    main()
