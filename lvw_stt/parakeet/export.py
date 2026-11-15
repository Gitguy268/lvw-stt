"""Merge the Parakeet LoRA adapter into dense Phonon-2.5 -> release/LvW-STT-EU-0.6B (fp32 safetensors, HF layout)."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoProcessor

from lvw_stt import trainlib
from lvw_stt.disk import ROOT, guard
from lvw_stt.langs import PHONON_LANGS
from lvw_stt.parakeet.expand_phonon import load_parakeet
from lvw_stt.paths import EU_NAME, PHONON_DENSE, RELEASE, RUNS, SRC_PHONON

NOTICE = """{name}
Copyright 2026 Lucas van de Weerd

This model is a fine-tune of Phonon-2.5 by Fermion Research
(https://huggingface.co/FermionResearch/Phonon-2.5, CC-BY-4.0), which is derived from
NVIDIA's Parakeet TDT 0.6B v3 (https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3, (c) NVIDIA
Corporation, CC-BY-4.0). The FastConformer encoder, TDT decoder, tokenizer and feature-extractor
configuration come from those models.

Changes: Phonon-2.5's five-value container was expanded to dense weights, then fine-tuned
(LoRA on the encoder attention and feed-forward layers, merged; full update of the decoder, joint
and encoder projector) on FLEURS train (CC-BY-4.0), Common Voice 22 train (CC0) and VoxPopuli
train (CC0). {extra}

These weights are distributed under CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/).
This is not an official Fermion Research or NVIDIA release.
"""


def transcribe(model, proc, rows):
    dev = trainlib.device()
    out = []
    for r in rows:
        w, _ = sf.read(ROOT / r["path"], dtype="float32")
        inp = proc([w], sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            s = model.generate(input_features=inp["input_features"].to(dev), attention_mask=inp["attention_mask"].to(dev))
        out.append(proc.batch_decode(getattr(s, "sequences", s), skip_special_tokens=True)[0])
    return out


def write_release(model, out: Path, name: str, extra: str = "") -> None:
    guard(2.6, what=f"{name} safetensors")
    shutil.rmtree(out, ignore_errors=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoProcessor.from_pretrained(PHONON_DENSE).save_pretrained(out)
    model.generation_config.save_pretrained(out)
    shutil.copy(SRC_PHONON / "LICENSE-WEIGHTS-CC-BY-4.0.txt", out / "LICENSE")
    (out / "NOTICE").write_text(NOTICE.format(name=name, extra=extra))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(RUNS / "parakeet-ft" / "best"))
    ap.add_argument("--out", default=str(RELEASE / EU_NAME))
    a = ap.parse_args()
    adapter = Path(a.adapter)
    if not (adapter / "adapter_model.safetensors").exists():
        print(f"no Parakeet adapter at {adapter}; skipped"); return
    dev = trainlib.device()
    proc = AutoProcessor.from_pretrained(PHONON_DENSE)
    peft = PeftModel.from_pretrained(load_parakeet(PHONON_DENSE), adapter).to(dev).eval()
    rows = trainlib.dev_rows(PHONON_LANGS, 1)[:12]
    before = transcribe(peft, proc, rows)
    merged = peft.merge_and_unload().eval()
    after = transcribe(merged, proc, rows)
    same = sum(x == y for x, y in zip(before, after))
    print(f"merge check: {same}/{len(rows)} identical transcripts")
    if same < len(rows) - 1:
        raise SystemExit("merge changed outputs")
    write_release(merged.cpu(), Path(a.out), EU_NAME)
    (Path(a.out) / "lvw_stt.json").write_text(json.dumps({"adapter": str(adapter.relative_to(ROOT)),
                                                          "base": "FermionResearch/Phonon-2.5@2af4bb8"}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
