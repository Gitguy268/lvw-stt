"""Fast variant: drop encoder layers from the fine-tuned model, then heal.

1. Score each encoder layer by block influence, 1 - cos(input, output), averaged over calibration clips (ShortGPT).
2. Remove the `--drop` least influential layers (never the first or last two).
3. Heal: full fine-tune of the kept encoder layers + decoder/joint/projector with TDT loss plus a masked MSE pulling
   the student's encoder output toward the unpruned teacher's. Adafactor keeps optimizer memory near zero.

  python -m lvw_stt.parakeet.prune --smoke
  python -m lvw_stt.parakeet.prune            # -> runs/parakeet-fast/, release/LvW-STT-EU-0.6B-Fast
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
from transformers import AutoProcessor, ParakeetForTDT

from lvw_stt import langs, trainlib
from lvw_stt.parakeet.export import write_release
from lvw_stt.parakeet.train import FPS, freeze_batchnorm, load_wave, make_eval, no_layerdrop, spec_augment
from lvw_stt.paths import EU_NAME, PHONON_DENSE, RELEASE, RUNS


@torch.no_grad()
def block_influence(model, proc, rows) -> list[float]:
    dev = trainlib.device()
    layers = model.encoder.layers
    acc = [0.0] * len(layers)
    hooks, cur = [], {}

    def mk(i):
        def hook(mod, args, kwargs, out):
            x = args[0] if args else kwargs["hidden_states"]
            y = out[0] if isinstance(out, tuple) else out
            cur[i] = 1 - F.cosine_similarity(x.float(), y.float(), dim=-1).mean().item()
        return hook
    for i, l in enumerate(layers):
        hooks.append(l.register_forward_hook(mk(i), with_kwargs=True))
    for r in rows:
        inp = proc([load_wave(r)], sampling_rate=16000, return_tensors="pt")
        model.get_audio_features(input_features=inp["input_features"].to(dev), attention_mask=inp["attention_mask"].to(dev))
        for i in cur:
            acc[i] += cur[i] / len(rows)
    for h in hooks:
        h.remove()
    return acc


def prune(model: ParakeetForTDT, drop: list[int]) -> ParakeetForTDT:
    keep = [i for i in range(len(model.encoder.layers)) if i not in drop]
    model.encoder.layers = torch.nn.ModuleList([model.encoder.layers[i] for i in keep])
    for new, l in enumerate(model.encoder.layers):
        if hasattr(l.self_attn, "layer_idx"):
            l.self_attn.layer_idx = new
    model.config.encoder_config.num_hidden_layers = len(keep)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=str(RELEASE / EU_NAME))
    ap.add_argument("--drop", type=int, default=6)
    ap.add_argument("--calib", type=int, default=120)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--kd", type=float, default=1.0, help="weight of the encoder-feature MSE")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=7000)
    ap.add_argument("--grad-acc", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default=str(RUNS / "parakeet-fast"))
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
    proc = AutoProcessor.from_pretrained(PHONON_DENSE)
    codes = langs.PHONON_LANGS

    teacher = ParakeetForTDT.from_pretrained(a.teacher, dtype=torch.bfloat16).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = ParakeetForTDT.from_pretrained(a.teacher, dtype=torch.float32)
    rows = trainlib.train_rows(codes, a.smoke, a.seed)

    plan = out / "prune_plan.json"
    if plan.exists():
        drop = json.loads(plan.read_text())["drop"]
    else:
        calib = random.Random(a.seed).sample(rows, min(a.calib, len(rows)))
        bi = block_influence(student.to(dev).eval(), proc, calib)
        n = len(bi)
        cand = sorted(range(2, n - 2), key=lambda i: bi[i])
        drop = sorted(cand[:a.drop])
        plan.write_text(json.dumps({"drop": drop, "block_influence": [round(x, 5) for x in bi]}, indent=1))
    print("dropping encoder layers", drop)
    student = no_layerdrop(prune(student.cpu(), drop))
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    freeze_batchnorm(student).to(dev).train()
    for n_, p in student.named_parameters():
        p.requires_grad_(not n_.startswith("encoder.subsampling"))
    params = [p for p in student.parameters() if p.requires_grad]
    print(f"student params {sum(p.numel() for p in student.parameters()):,}, trainable {sum(p.numel() for p in params):,}")

    tok = proc.tokenizer
    ntok = {r["id"]: len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows}
    cost = lambda r: math.ceil(r["dur"] * FPS) * (ntok[r["id"]] + 1)
    batches_for_epoch = lambda e: trainlib.buckets(rows, cost, a.budget, 16, a.seed + e)

    def step(batch):
        inp = proc([load_wave(r) for r in batch], text=[r["text"] for r in batch], sampling_rate=16000,
                   return_tensors="pt", padding=True)
        feats = inp["input_features"].to(dev); am = inp["attention_mask"].to(dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            t_enc = teacher.get_audio_features(input_features=feats, attention_mask=am).last_hidden_state.float()
        if student.training and not a.smoke:
            feats = spec_augment(feats)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            s = student.get_audio_features(input_features=feats, attention_mask=am)
            o = student(encoder_outputs=s, decoder_input_ids=inp["decoder_input_ids"].to(dev), labels=inp["labels"].to(dev))
        m = s.attention_mask.unsqueeze(-1).float()
        mse = (((s.last_hidden_state.float() - t_enc) ** 2) * m).sum() / (m.sum() * t_enc.shape[-1])
        return o.loss + a.kd * mse / t_enc.var().clamp_min(1e-6)

    # bf16 on disk (~1 GB each) and keep=1: three fp32 copies would not fit next to the training audio
    save = lambda d: (d.mkdir(parents=True, exist_ok=True),
                      torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v
                                  for k, v in student.state_dict().items()}, d / "student.pt"))
    load = lambda d: student.load_state_dict({k: v.float() if v.is_floating_point() else v
                                              for k, v in torch.load(d / "student.pt", weights_only=True).items()})

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
                          make_eval(student, proc, codes, 10), save, load)
    load(out / "best")
    keep = len(student.encoder.layers)
    write_release(student.cpu().eval(), RELEASE / f"{EU_NAME}-Fast", f"{EU_NAME}-Fast",
                  extra=f"Encoder pruned from 24 to {keep} layers (dropped {drop}) and healed with TDT loss plus "
                        f"encoder-feature distillation from {EU_NAME}.")
    print(f"done: best dev {state['best']:.3f}; wrote {RELEASE / (EU_NAME + '-Fast')}")


if __name__ == "__main__":
    main()
