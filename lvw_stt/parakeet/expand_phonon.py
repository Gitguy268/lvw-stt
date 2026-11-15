"""Phonon-2.5 five-value container -> dense fp32 transformers ParakeetForTDT.

The dense weights are never written to disk (2.5 GB): load_parakeet() rebuilds them from the 178 MB container in
seconds. runs/phonon25-dense/ only holds config, processor and generation config, so it works as a model "path".

Follows FermionResearch's reference_transformers.container_state_dict(): pointwise convs back to [O, I, 1],
fp16-stored BatchNorm counters restored to int64 (0 when non-finite; eval never reads them).
"""
from __future__ import annotations

import json
import re

import numpy as np
import torch
from transformers import AutoProcessor, GenerationConfig, ParakeetForTDT, ParakeetTDTConfig

from lvw_stt.fetch import phonon_container
from lvw_stt.parakeet.fermion_container import read_container
from lvw_stt.paths import PHONON_DENSE, SRC_PARAKEET_CFG


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


def build() -> ParakeetForTDT:
    cfg = ParakeetTDTConfig.from_pretrained(SRC_PARAKEET_CFG)
    model = ParakeetForTDT(cfg)
    sd, index = container_state_dict(phonon_container())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"weight load: missing {missing[:5]} unexpected {unexpected[:5]}")
    print(f"container records {len(index)}, state dict keys {len(sd)}, params {sum(p.numel() for p in model.parameters()):,}")
    model.generation_config = GenerationConfig.from_pretrained(SRC_PARAKEET_CFG)
    return model.eval()


def load_parakeet(path, dtype=torch.float32) -> ParakeetForTDT:
    """A model directory, or PHONON_DENSE -> rebuilt from the Phonon container."""
    from pathlib import Path
    if Path(path).resolve() == PHONON_DENSE.resolve():
        return build().to(dtype)
    return ParakeetForTDT.from_pretrained(path, dtype=dtype).eval()


def check(model: ParakeetForTDT, n: int = 25) -> None:
    """Transcribe a few FLEURS dev clips: the expansion must give sane text (WER well under 30%)."""
    import soundfile as sf
    from lvw_stt.data import DATA, read_jsonl
    from lvw_stt.disk import ROOT
    from lvw_stt.textnorm import score
    rows = [r for r in read_jsonl(DATA / "eval" / "fleurs-dev.jsonl") if r["lang"] in ("en", "nl", "de", "el", "pl")]
    rows = rows[:n]
    if not rows:
        print("no fleurs-dev clips yet (bash run.sh prepare-eval); skipped transcript check")
        return
    proc = AutoProcessor.from_pretrained(PHONON_DENSE)
    refs, hyps = [], []
    for r in rows:
        w, _ = sf.read(ROOT / r["path"], dtype="float32")
        inp = proc([w], sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            seq = model.generate(input_features=inp["input_features"], attention_mask=inp["attention_mask"])
        t = proc.batch_decode(getattr(seq, "sequences", seq), skip_special_tokens=True)[0]
        refs.append(r["text"]); hyps.append(t)
        print(f"  {r['lang']} {t[:90]}")
    s = score(refs, hyps, "xx")
    print(f"expansion check: WER {s['err']:.1f}% on {s['n']} clips (basic normaliser, mixed languages)")
    if s["err"] > 30:
        raise SystemExit("expanded Phonon transcribes badly: container/config mismatch")


def main() -> None:
    PHONON_DENSE.mkdir(parents=True, exist_ok=True)
    model = build()
    model.config.save_pretrained(PHONON_DENSE)
    model.generation_config.save_pretrained(PHONON_DENSE)
    AutoProcessor.from_pretrained(SRC_PARAKEET_CFG).save_pretrained(PHONON_DENSE)
    (PHONON_DENSE / "SOURCE.json").write_text(json.dumps({
        "derived_from": "FermionResearch/Phonon-2.5@2af4bb8ccceb66a8ce06c4081bb968f1802c1e58",
        "base": "nvidia/parakeet-tdt-0.6b-v3@541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        "license": "cc-by-4.0", "weights": "rebuilt from the container by load_parakeet()"}, indent=1))
    check(model)


if __name__ == "__main__":
    main()
