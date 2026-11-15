"""Fast variant of the fine-tuned Qwen3-ASR: remove a contiguous block of decoder layers, then heal.

1. Pick the block [l, l+n) whose input and output hidden states are closest in angle on calibration clips
   (Gromov et al., "The Unreasonable Ineffectiveness of the Deeper Layers"); first and last layer always stay.
2. Heal: full fine-tune of the remaining decoder layers + final norm (audio tower, embeddings and tied head frozen),
   cross-entropy plus KL to the unpruned fine-tuned teacher on the transcript tokens. Adafactor, bf16 autocast.

  python -m lvw_stt.qwen.prune --smoke
  python -m lvw_stt.qwen.prune             # -> runs/qwen-fast/, release/LvW-STT-Qwen3-0.6B-Fast
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from lvw_stt import langs, trainlib
from lvw_stt.qwen.export import load_wrapper, write_release
from lvw_stt.qwen.train import label_mask, load_wave, make_eval, prefix_text, target
from lvw_stt.paths import QW_NAME, RELEASE, RUNS


def encode(proc, batch, dev, dtype):
    pfx = prefix_text(proc)
    eos = proc.tokenizer.eos_token or ""
    proc.tokenizer.padding_side = "right"
    audios = [load_wave(r) for r in batch]
    full = proc(text=[pfx + target(r) + eos for r in batch], audio=audios, return_tensors="pt", padding=True)
    labels = label_mask(proc, pfx, audios, full)
    inputs = {k: (v.to(dev, dtype) if v.is_floating_point() else v.to(dev)) for k, v in full.items()}
    return inputs, labels.to(dev)


@torch.no_grad()
def choose_block(thinker, proc, rows, n: int) -> tuple[int, list[float]]:
    dev = trainlib.device()
    L = len(thinker.model.layers)
    dist = [0.0] * (L - n + 1)
    for r in rows:
        inputs, labels = encode(proc, [r], dev, torch.bfloat16)
        hs = thinker(**inputs, output_hidden_states=True).hidden_states   # L+1 tensors
        m = labels[0] != -100
        for l in range(L - n + 1):
            a, b = hs[l][0][m].float(), hs[l + n][0][m].float()
            dist[l] += (torch.arccos(F.cosine_similarity(a, b, dim=-1).clamp(-1, 1)) / math.pi).mean().item() / len(rows)
    cand = [l for l in range(1, L - n) if l + n <= L - 1]
    return min(cand, key=lambda l: dist[l]), dist


def prune(model, start: int, n: int):
    text = model.thinker.model
    keep = [i for i in range(len(text.layers)) if not (start <= i < start + n)]
    text.layers = torch.nn.ModuleList([text.layers[i] for i in keep])
    for new, l in enumerate(text.layers):
        l.self_attn.layer_idx = new
    for cfg in (model.config.thinker_config.text_config, text.config):
        cfg.num_hidden_layers = len(keep)
        if isinstance(getattr(cfg, "layer_types", None), list):
            cfg.layer_types = [cfg.layer_types[i] for i in keep]
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=str(RELEASE / QW_NAME))
    ap.add_argument("--drop", type=int, default=8)
    ap.add_argument("--calib", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--budget-s", type=float, default=30.0)
    ap.add_argument("--grad-acc", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default=str(RUNS / "qwen-fast"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = trainlib.device()
    out = Path(a.out + ("-smoke" if a.smoke else ""))
    if a.smoke:
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    codes = langs.QWEN_LANGS

    teacher = load_wrapper(a.teacher).model.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    sw = load_wrapper(a.teacher)
    student, proc = sw.model, sw.processor
    rows = trainlib.train_rows(codes, a.smoke, a.seed)

    plan = out / "prune_plan.json"
    if plan.exists():
        start = json.loads(plan.read_text())["start"]
    else:
        calib = random.Random(a.seed).sample(rows, min(a.calib, len(rows)))
        start, dist = choose_block(teacher.thinker, proc, calib, a.drop)
        plan.write_text(json.dumps({"start": start, "n": a.drop, "angular_distance": [round(x, 5) for x in dist]}, indent=1))
    print(f"dropping decoder layers {start}..{start + a.drop - 1}")
    student = prune(student.float(), start, a.drop)
    th = student.thinker
    th.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for n_, p in student.named_parameters():
        p.requires_grad_(".model.layers." in n_ or n_.endswith("model.norm.weight"))
    params = [p for p in student.parameters() if p.requires_grad]
    student.to(dev).train()
    print(f"student params {sum(p.numel() for p in student.parameters()):,}, trainable {sum(p.numel() for p in params):,}")
    batches_for_epoch = lambda e: trainlib.buckets(rows, lambda r: r["dur"], a.budget_s, 4, a.seed + e)

    def step(batch):
        inputs, labels = encode(proc, batch, dev, torch.float32)
        with torch.no_grad():
            t_logits = teacher.thinker(**{k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                                          for k, v in inputs.items()}).logits
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            o = th(**inputs, labels=labels)
        # logits at position t predict token t+1: shift as the CE does
        m = labels[:, 1:] != -100
        s_l = o.logits[:, :-1][m].float()
        t_l = t_logits[:, :-1][m].float()
        kl = F.kl_div(F.log_softmax(s_l, -1), F.log_softmax(t_l, -1), log_target=True, reduction="batchmean")
        return o.loss + a.kd * kl

    trainable_names = [n_ for n_, p in student.named_parameters() if p.requires_grad]

    def save(d: Path):
        d.mkdir(parents=True, exist_ok=True)
        sd = student.state_dict()
        torch.save({k: sd[k].to(torch.bfloat16) for k in trainable_names}, d / "student.pt")

    def load(d: Path):
        sd = torch.load(d / "student.pt", weights_only=True)
        student.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)

    if a.smoke:
        fixed = batches_for_epoch(0)[0]
        student.eval()
        with torch.no_grad():
            before = float(step(fixed))
        cfg = trainlib.Cfg(out=out, epochs=1000, grad_acc=1, warmup=5, max_steps=30, smoke=True, log_every=5,
                           optim="adafactor")
        state = trainlib.loop(cfg, student, [{"params": params, "lr": a.lr * 5}], batches_for_epoch, step, lambda: 0.0,
                              save, load)
        student.eval()
        with torch.no_grad():
            after = float(step(fixed))
        save(out / "rt"); load(out / "rt")
        with torch.no_grad():
            reloaded = float(step(fixed))
        raise SystemExit(0 if trainlib.smoke_report(before, after, reloaded, state["peak_gb"], rel_tol=2e-2) else 1)

    cfg = trainlib.Cfg(out=out, epochs=a.epochs, grad_acc=a.grad_acc, eval_every=a.eval_every, save_every=a.eval_every,
                       resume=a.resume, seed=a.seed, optim="adafactor", keep=1)
    state = trainlib.loop(cfg, student, [{"params": params, "lr": a.lr}], batches_for_epoch, step,
                          make_eval(student, proc, codes, 8), save, load)
    load(out / "best")
    write_release(student.to(torch.bfloat16).cpu().eval(), RELEASE / f"{QW_NAME}-Fast", f"{QW_NAME}-Fast",
                  extra=f"Decoder pruned from 28 to {28 - a.drop} layers (removed {start}..{start + a.drop - 1}) and "
                        f"healed with cross-entropy plus KL distillation from {QW_NAME}.")
    print(f"done: best dev {state['best']:.3f}; wrote {RELEASE / (QW_NAME + '-Fast')}")


if __name__ == "__main__":
    main()
