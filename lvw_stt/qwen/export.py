"""Merge the Qwen3-ASR LoRA adapter -> release/LvW-STT-Qwen3-0.6B in the original repo layout (loads with qwen-asr)."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel

from lvw_stt import trainlib
from lvw_stt.disk import ROOT, guard
from lvw_stt.langs import QWEN_LANGS
from lvw_stt.paths import QW_NAME, RELEASE, RUNS, SRC_QWEN

SIDE_FILES = ["generation_config.json", "preprocessor_config.json", "processor_config.json", "tokenizer_config.json",
              "tokenizer.json", "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json"]

NOTICE = """{name}
Copyright 2026 Lucas van de Weerd

Fine-tune of Qwen/Qwen3-ASR-0.6B (https://huggingface.co/Qwen/Qwen3-ASR-0.6B, Apache-2.0, (c) Alibaba Qwen team).
Changes: LoRA on the text decoder (attention + MLP) and audio-encoder attention, merged into the weights, trained on
FLEURS train (CC-BY-4.0), Common Voice 22 train (CC0) and VoxPopuli train (CC0). {extra}
Licensed under the Apache License, Version 2.0. Not an official Qwen release.
"""


def load_wrapper(path, device_map=None):
    from qwen_asr import Qwen3ASRModel
    return Qwen3ASRModel.from_pretrained(str(path), dtype=torch.bfloat16, device_map=device_map,
                                         max_inference_batch_size=8, max_new_tokens=256)


def transcribe(wrapper, rows):
    wrapper.processor.tokenizer.padding_side = "left"
    res = wrapper.transcribe(audio=[(sf.read(ROOT / r["path"], dtype="float32")[0], 16000) for r in rows], language=None)
    return [r.text for r in res]


def write_release(model, out: Path, name: str, extra: str = "") -> None:
    guard(2.0, what=f"{name} safetensors")
    shutil.rmtree(out, ignore_errors=True)
    model.save_pretrained(out, safe_serialization=True)
    for f in SIDE_FILES:
        if (SRC_QWEN / f).exists():
            shutil.copy2(SRC_QWEN / f, out / f)
    shutil.copy2(ROOT / "third_party" / "Qwen3-ASR" / "LICENSE", out / "LICENSE")
    (out / "NOTICE").write_text(NOTICE.format(name=name, extra=extra))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(RUNS / "qwen-ft" / "best"))
    ap.add_argument("--out", default=str(RELEASE / QW_NAME))
    a = ap.parse_args()
    adapter = Path(a.adapter)
    if not (adapter / "adapter_model.safetensors").exists():
        print(f"no Qwen adapter at {adapter}; skipped"); return
    wrapper = load_wrapper(SRC_QWEN)
    wrapper.model.thinker = PeftModel.from_pretrained(wrapper.model.thinker, adapter)
    wrapper.model.to(trainlib.device()).eval()
    wrapper.device = wrapper.model.device
    rows = trainlib.dev_rows(QWEN_LANGS, 1)[:12]
    before = transcribe(wrapper, rows)
    wrapper.model.thinker = wrapper.model.thinker.merge_and_unload()
    after = transcribe(wrapper, rows)
    same = sum(x == y for x, y in zip(before, after))
    print(f"merge check: {same}/{len(rows)} identical transcripts")
    if same < len(rows) - 2:          # bf16 merge can flip a near-tie token
        raise SystemExit("merge changed outputs")
    write_release(wrapper.model.cpu(), Path(a.out), QW_NAME)
    reloaded = load_wrapper(a.out, device_map=str(trainlib.device()))
    print("reload:", transcribe(reloaded, rows[:2]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
