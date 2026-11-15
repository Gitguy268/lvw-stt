"""Phonon five-value container -> dense fp32 transformers ParakeetForTDT.

The dense weights are never written to disk (2.5 GB): load_parakeet() rebuilds them from the 178 MB container in
seconds. A dense dir (runs/phonon25-dense = Phonon-2.5, runs/phonon25-en-dense = Phonon-2.5-English) only holds
config, processor, generation config and SOURCE.json, which names the container (and its sha256) to rebuild from.

  python -m lvw_stt.parakeet.expand_phonon                 # Phonon-2.5 (25 languages)
  python -m lvw_stt.parakeet.expand_phonon --source en     # Phonon-2.5-English

Follows FermionResearch's reference_transformers.container_state_dict(): pointwise convs back to [O, I, 1],
fp16-stored BatchNorm counters restored to int64 (0 when non-finite; eval never reads them).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, GenerationConfig, ParakeetForTDT, ParakeetTDTConfig

from lvw_stt.disk import ROOT
from lvw_stt.fetch import phonon_container, sha256
from lvw_stt.parakeet.fermion_container import read_container
from lvw_stt.paths import PHONON_DENSE, PHONON_SOURCES, SRC_PARAKEET_CFG


def container_state_dict(path) -> tuple[dict, list]:
    tensors, index = read_container(str(path), with_raw=False)
    sd = {}
    for k, v in tensors.items():
        if k.endswith("num_batches_tracked"):
            v32 = float(np.asarray(v, dtype=np.float32).reshape(-1)[0])
            sd[k] = torch.tensor(int(v32) if np.isfinite(v32) else 0, dtype=torch.int64)
            continue
        arr = np.ascontiguousarray(v.astype(np.float32))
        if arr.ndim == 2 and re.search(r"\.conv\.pointwise_conv[12]\.weight$", k):
            arr = arr[:, :, None]
        sd[k] = torch.from_numpy(arr)
    return sd, index


def build(container: Path) -> ParakeetForTDT:
    cfg = ParakeetTDTConfig.from_pretrained(SRC_PARAKEET_CFG)
    model = ParakeetForTDT(cfg)
    sd, index = container_state_dict(container)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"weight load: missing {missing[:5]} unexpected {unexpected[:5]}")
    print(f"{container.parent.parent.name}: container records {len(index)}, state dict keys {len(sd)}, "
          f"params {sum(p.numel() for p in model.parameters()):,}")
    model.generation_config = GenerationConfig.from_pretrained(SRC_PARAKEET_CFG)
    return model.eval()


def source_container(dense_dir: Path) -> Path | None:
    """The container a dense dir is rebuilt from, checksum-verified; None for a normal model directory."""
    meta = dense_dir / "SOURCE.json"
    if not meta.exists() or any(dense_dir.glob("*.safetensors")):
        return None
    info = json.loads(meta.read_text())
    if "container" in info:
        cont, want = ROOT / info["container"], info["container_sha256"]
    elif dense_dir.resolve() == PHONON_DENSE.resolve():       # written before SOURCE.json named its container
        cont, want = phonon_container(PHONON_SOURCES["multi"]["src"]), None
    else:
        raise SystemExit(f"{meta}: no container recorded; re-run expand_phonon")
    if want and sha256(cont) != want:
        raise SystemExit(f"{cont}: checksum differs from {meta}")
    return cont


def load_parakeet(path, dtype=torch.float32) -> ParakeetForTDT:
    """A model directory, or a dense dir from this module -> rebuilt from its Phonon container."""
    cont = source_container(Path(path))
    if cont is not None:
        return build(cont).to(dtype)
    return ParakeetForTDT.from_pretrained(path, dtype=dtype).eval()


def check(model: ParakeetForTDT, dense_dir: Path, codes, max_wer: float, n: int = 25) -> None:
    """Transcribe a few FLEURS dev clips: the expansion must give sane text."""
    import soundfile as sf
    from lvw_stt.data import DATA, read_jsonl
    from lvw_stt.textnorm import score
    rows = [r for r in read_jsonl(DATA / "eval" / "fleurs-dev.jsonl") if r["lang"] in codes][:n]
    if not rows:
        print("no fleurs-dev clips yet (bash run.sh prepare-eval); skipped transcript check")
        return
    proc = AutoProcessor.from_pretrained(dense_dir)
    refs, hyps = [], []
    for r in rows:
        w, _ = sf.read(ROOT / r["path"], dtype="float32")
        inp = proc([w], sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            seq = model.generate(input_features=inp["input_features"], attention_mask=inp["attention_mask"])
        t = proc.batch_decode(getattr(seq, "sequences", seq), skip_special_tokens=True)[0]
        refs.append(r["text"]); hyps.append(t)
        print(f"  {r['lang']} {t[:90]}")
    s = score(refs, hyps, codes[0] if len(codes) == 1 else "xx")
    print(f"expansion check: WER {s['err']:.1f}% on {s['n']} clips ({', '.join(codes)})")
    if s["err"] > max_wer:
        raise SystemExit(f"expanded Phonon transcribes badly (> {max_wer}%): container/config mismatch")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(PHONON_SOURCES), default="multi")
    a = ap.parse_args()
    s = PHONON_SOURCES[a.source]
    dense: Path = s["dense"]
    cont = phonon_container(s["src"])
    model = build(cont)
    dense.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(dense)
    model.generation_config.save_pretrained(dense)
    AutoProcessor.from_pretrained(SRC_PARAKEET_CFG).save_pretrained(dense)
    (dense / "SOURCE.json").write_text(json.dumps({
        "derived_from": f"{s['repo']}@{s['rev']}",
        "base": "nvidia/parakeet-tdt-0.6b-v3@541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        "container": str(cont.relative_to(ROOT)), "container_sha256": sha256(cont),
        "license": "cc-by-4.0", "weights": "rebuilt from the container by load_parakeet()"}, indent=1))
    check(model, dense, list(s["check_langs"]), s["check_max_wer"])


if __name__ == "__main__":
    main()
