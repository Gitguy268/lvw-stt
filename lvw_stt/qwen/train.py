"""LoRA fine-tune of Qwen3-ASR-0.6B on its 30 languages, on Apple silicon.

Prompt/label layout is the one in QwenLM/Qwen3-ASR finetuning/qwen3_asr_sft.py: chat-template prefix (empty system,
user audio) + "language <Name><asr_text><text>" + eos, with the prefix masked out of the loss. That script has no LoRA
and assumes CUDA for bf16, so this wraps the same layout with PEFT and runs bf16 on MPS.

  python -m lvw_stt.qwen.train --smoke
  python -m lvw_stt.qwen.train                  # -> runs/qwen-ft/
  python -m lvw_stt.qwen.train --resume
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

from lvw_stt import langs, trainlib
from lvw_stt.disk import ROOT
from lvw_stt.paths import RUNS, SRC_QWEN
from lvw_stt.textnorm import score

LORA_TARGETS = (r".*model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
                r"|.*audio_tower\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)")


def load_wave(r):
    w, _ = sf.read(ROOT / r["path"], dtype="float32")
    return w


def target(r) -> str:
    return f"language {langs.qwen_name(r['lang'])}<asr_text>{r['text']}"


def build(args):
    from qwen_asr import Qwen3ASRModel
    wrapper = Qwen3ASRModel.from_pretrained(str(args.init or SRC_QWEN), dtype=torch.bfloat16, device_map=None,
                                            max_inference_batch_size=8, max_new_tokens=256)
    model, proc = wrapper.model, wrapper.processor
    thinker = model.thinker
    if args.grad_ckpt:
        thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none")
    peft_thinker = get_peft_model(thinker, lcfg)        # injects LoRA into model.thinker's modules in place
    peft_thinker.print_trainable_parameters()
    model.to(trainlib.device())
    return model, peft_thinker, proc


def prefix_text(proc) -> str:
    msgs = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": None}]}]
    return proc.apply_chat_template([msgs], add_generation_prompt=True, tokenize=False)[0]


def label_mask(proc, pfx: str, audios, full) -> "torch.Tensor":
    """Labels = input ids with everything up to and including the prompt masked. The prompt length depends on the
    clip (audio tokens), and encoding all prefixes as one padded batch gives wrong lengths (loss 7.7 instead of 0.5),
    so each prefix is measured on its own; works for left or right padding."""
    labels = full["input_ids"].clone()
    for i, w in enumerate(audios):
        n = int(proc(text=[pfx], audio=[w], return_tensors="pt")["attention_mask"].sum())
        start = int(full["attention_mask"][i].nonzero()[0])
        labels[i, :start + n] = -100
    labels[full["attention_mask"] == 0] = -100
    return labels


def make_step(peft_thinker, proc, args):
    dev = trainlib.device()
    pfx = prefix_text(proc)
    eos = proc.tokenizer.eos_token or ""

    def step(batch):
        proc.tokenizer.padding_side = "right"
        audios = [load_wave(r) for r in batch]
        full = proc(text=[pfx + target(r) + eos for r in batch], audio=audios, return_tensors="pt", padding=True)
        labels = label_mask(proc, pfx, audios, full)
        inputs = {k: (v.to(dev, torch.bfloat16) if v.is_floating_point() else v.to(dev)) for k, v in full.items()}
        out = peft_thinker(**inputs, labels=labels.to(dev))
        return out.loss
    return step


def make_eval(model, proc, codes, per_lang):
    from qwen_asr import Qwen3ASRModel
    rows = trainlib.dev_rows(codes, per_lang)
    asr = Qwen3ASRModel(backend="transformers", model=model, processor=proc, max_inference_batch_size=8, max_new_tokens=256)

    def ev():
        proc.tokenizer.padding_side = "left"
        refs, hyps = {}, {}
        res = asr.transcribe(audio=[(load_wave(r), 16000) for r in rows], language=None)
        for r, t in zip(rows, res):
            refs.setdefault(r["lang"], []).append(r["text"]); hyps.setdefault(r["lang"], []).append(t.text)
        errs = [score(refs[c], hyps[c], c)["err"] for c in refs]
        errs = [e for e in errs if not math.isnan(e)]
        return sum(errs) / max(1, len(errs))
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS / "qwen-ft"))
    ap.add_argument("--init")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--budget-s", type=float, default=40.0, help="audio seconds per micro-batch (160 swapped on 16 GB)")
    ap.add_argument("--max-batch", type=int, default=4)
    ap.add_argument("--grad-acc", type=int, default=16)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--dev-per-lang", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    codes = langs.QWEN_LANGS
    model, peft_thinker, proc = build(args)
    rows = trainlib.train_rows(codes, args.smoke, args.seed)
    batches_for_epoch = lambda e: trainlib.buckets(rows, lambda r: r["dur"], args.budget_s, args.max_batch, args.seed + e)
    params = [p for p in peft_thinker.parameters() if p.requires_grad]
    groups = [{"params": params, "lr": args.lr}]
    print(f"{len(rows)} clips, {sum(r['dur'] for r in rows) / 3600:.1f} h; {len(batches_for_epoch(0))} micro-batches/epoch")

    step = make_step(peft_thinker, proc, args)
    out = Path(args.out)
    save = lambda d: peft_thinker.save_pretrained(d)
    load = lambda d: set_peft_model_state_dict(peft_thinker, load_file(d / "adapter_model.safetensors"))

    if args.smoke:
        out = out.with_name(out.name + "-smoke"); shutil.rmtree(out, ignore_errors=True)
        fixed = batches_for_epoch(0)[0]
        peft_thinker.eval()
        with torch.no_grad():
            before = float(step(fixed))
        cfg = trainlib.Cfg(out=out, epochs=1000, grad_acc=1, warmup=5, max_steps=50, smoke=True, log_every=5)
        state = trainlib.loop(cfg, peft_thinker, groups, batches_for_epoch, step, lambda: 0.0, save, load)
        peft_thinker.eval()
        with torch.no_grad():
            after = float(step(fixed))
        save(out / "rt")
        del model, peft_thinker
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        m2, pt2, _ = build(args); load_fn = lambda d: set_peft_model_state_dict(pt2, load_file(d / "adapter_model.safetensors"))
        load_fn(out / "rt"); pt2.eval()
        with torch.no_grad():
            reloaded = float(make_step(pt2, proc, args)(fixed))
        ok = trainlib.smoke_report(before, after, reloaded, state["peak_gb"])
        raise SystemExit(0 if ok else 1)

    cfg = trainlib.Cfg(out=out, epochs=args.epochs, grad_acc=args.grad_acc, eval_every=args.eval_every,
                       save_every=args.eval_every, resume=args.resume, seed=args.seed)
    state = trainlib.loop(cfg, peft_thinker, groups, batches_for_epoch, step,
                          make_eval(model, proc, codes, args.dev_per_lang), save, load)
    print(f"done: best dev {state['best']:.3f} at step {state['best_step']}; adapter in {out / 'best'}")


if __name__ == "__main__":
    main()
