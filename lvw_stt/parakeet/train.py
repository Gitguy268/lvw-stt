"""Fine-tune a dense Phonon (ParakeetForTDT): Phonon-2.5 on the 25 EU languages (eu profile) or Phonon-2.5-English
on English + Dutch (LVW_PROFILE=en-nl).

LoRA on every encoder attention / feed-forward projection, full training of the LSTM decoder, joint and encoder
projector, TDT loss (transformers' pure-PyTorch implementation, works on MPS). Batches are packed by the size of the
joint tensor (frames x tokens), which is what bounds memory.

  python -m lvw_stt.parakeet.train --smoke          # 50 steps on 64 clips: loss falls, save/load round-trip, memory
  python -m lvw_stt.parakeet.train                  # full run -> runs/parakeet-ft/
  python -m lvw_stt.parakeet.train --resume
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
from transformers import AutoProcessor

from lvw_stt import langs, trainlib
from lvw_stt.disk import ROOT
from lvw_stt.parakeet.expand_phonon import load_parakeet
from lvw_stt.paths import BASE_DENSE, PROFILE, PROFILE_LANGS, PROFILE_RUNS
from lvw_stt.textnorm import score

LORA_TARGETS = r".*encoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|feed_forward[12]\.linear[12])"
FULL_MODULES = ["decoder", "joint", "encoder_projector"]
FPS = 12.5   # encoder frames per second after 8x subsampling of 10 ms features


def spec_augment(x: torch.Tensor, n_freq=2, f_w=27, n_time=10, t_frac=0.05) -> torch.Tensor:
    """x: (B, T, F). In-place-free masking."""
    B, T, F = x.shape
    m = torch.ones_like(x)
    for b in range(B):
        for _ in range(n_freq):
            w = int(torch.randint(0, f_w + 1, ()).item()); f0 = int(torch.randint(0, max(1, F - w), ()).item())
            m[b, :, f0:f0 + w] = 0
        tw = max(1, int(T * t_frac))
        for _ in range(n_time):
            w = int(torch.randint(0, tw + 1, ()).item()); t0 = int(torch.randint(0, max(1, T - w), ()).item())
            m[b, t0:t0 + w, :] = 0
    return x * m


def freeze_batchnorm(model):
    """Keep BatchNorm on its stored statistics while training. Batch stats from a few padded clips are garbage
    (losses of ~400 instead of ~1), and updating the running stats would also change the model outside the adapter."""
    orig = model.train

    def train(mode: bool = True):
        orig(mode)
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
        return model
    model.train = train
    return model


def no_layerdrop(model):
    """The checkpoint config has layerdrop 0.1: in train mode each encoder layer is skipped 10% of the time.
    Phonon was never trained for that (batches hit losses of ~2000), and with gradient checkpointing the recompute
    can skip a different layer than the forward pass. Off, in the module and in the saved config."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.config.encoder_config.layerdrop = 0.0
    base.encoder.layerdrop = 0.0
    return model


def load_wave(r: dict):
    w, _ = sf.read(ROOT / r["path"], dtype="float32", always_2d=False)
    return w


def build(args):
    proc = AutoProcessor.from_pretrained(BASE_DENSE)
    model = load_parakeet(args.init or BASE_DENSE, dtype=torch.float32)
    no_layerdrop(model)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=LORA_TARGETS,
                      modules_to_save=FULL_MODULES, bias="none")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return freeze_batchnorm(model).to(trainlib.device()), proc


def make_step(model, proc, args):
    dev = trainlib.device()
    fdim = proc.feature_extractor.feature_size

    def step(batch):
        inp = proc([load_wave(r) for r in batch], text=[r["text"] for r in batch], sampling_rate=16000,
                   return_tensors="pt", padding=True)
        feats = inp["input_features"]
        assert feats.shape[-1] == fdim, feats.shape
        if args.specaug and model.training:
            feats = spec_augment(feats)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=args.amp == "bf16"):
            out = model(input_features=feats.to(dev), attention_mask=inp["attention_mask"].to(dev),
                        decoder_input_ids=inp["decoder_input_ids"].to(dev), labels=inp["labels"].to(dev))
        return out.loss
    return step


def make_eval(model, proc, codes, per_lang):
    rows = trainlib.dev_rows(codes, per_lang)
    dev = trainlib.device()

    def ev():
        refs, hyps = {}, {}
        for i in range(0, len(rows), 8):
            b = rows[i:i + 8]
            inp = proc([load_wave(r) for r in b], sampling_rate=16000, return_tensors="pt", padding=True)
            seq = model.generate(input_features=inp["input_features"].to(dev), attention_mask=inp["attention_mask"].to(dev))
            txt = proc.batch_decode(getattr(seq, "sequences", seq), skip_special_tokens=True)
            for r, t in zip(b, txt):
                g = trainlib.dev_group(r)
                refs.setdefault(g, []).append(r["text"]); hyps.setdefault(g, []).append(t)
        errs = [score(refs[g], hyps[g], g.split("/")[0])["err"] for g in refs]
        errs = [e for e in errs if not math.isnan(e)]
        return sum(errs) / max(1, len(errs))
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROFILE_RUNS / "parakeet-ft"))
    ap.add_argument("--init", help="start weights (default: the profile's dense Phonon)")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-full", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--budget", type=float, default=8000, help="max sum of frames x (tokens+1) per micro-batch")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--grad-acc", type=int, default=4)
    ap.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    ap.add_argument("--no-specaug", dest="specaug", action="store_false")
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--dev-per-lang", type=int, default=60 if PROFILE == "en-nl" else 10,
                    help="dev clips per group (language, or language+source in en-nl)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    codes = PROFILE_LANGS or langs.PHONON_LANGS
    print(f"profile {PROFILE}: base {BASE_DENSE.name}, languages {codes}")
    model, proc = build(args)
    rows = trainlib.train_rows(codes, args.smoke, args.seed)
    tok = proc.tokenizer
    ntok = {r["id"]: len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows}
    cost = lambda r: math.ceil(r["dur"] * FPS) * (ntok[r["id"]] + 1)
    batches_for_epoch = lambda e: trainlib.buckets(rows, cost, args.budget, args.max_batch, args.seed + e)

    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    full = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    groups = [{"params": lora, "lr": args.lr_lora}, {"params": full, "lr": args.lr_full}]
    print(f"{len(rows)} clips, {sum(r['dur'] for r in rows) / 3600:.1f} h; {len(batches_for_epoch(0))} micro-batches/epoch")

    step = make_step(model, proc, args)
    out = Path(args.out)
    save = lambda d: model.save_pretrained(d)

    def load(d: Path):
        set_peft_model_state_dict(model, load_file(d / "adapter_model.safetensors"))

    if args.smoke:
        out = out.with_name(out.name + "-smoke"); shutil.rmtree(out, ignore_errors=True)
        args.specaug = False                                   # deterministic fixed-batch loss
        fixed = batches_for_epoch(0)[0]
        model.eval()
        with torch.no_grad():
            before = float(step(fixed))
        cfg = trainlib.Cfg(out=out, epochs=1000, grad_acc=1, warmup=5, max_steps=50, smoke=True, log_every=5)
        state = trainlib.loop(cfg, model, groups, batches_for_epoch, step, lambda: 0.0, save, load)
        model.eval()
        with torch.no_grad():
            after = float(step(fixed))
        save(out / "rt")
        del model, groups, lora, full
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        m2, _ = build(args); set_peft_model_state_dict(m2, load_file(out / "rt" / "adapter_model.safetensors"))
        m2.eval()
        with torch.no_grad():
            reloaded = float(make_step(m2, proc, args)(fixed))
        ok = trainlib.smoke_report(before, after, reloaded, state["peak_gb"])
        raise SystemExit(0 if ok else 1)

    cfg = trainlib.Cfg(out=out, epochs=args.epochs, grad_acc=args.grad_acc, eval_every=args.eval_every,
                       save_every=args.eval_every, resume=args.resume, seed=args.seed)
    state = trainlib.loop(cfg, model, groups, batches_for_epoch, step, make_eval(model, proc, codes, args.dev_per_lang),
                          save, load)
    print(f"done: best dev {state['best']:.3f} at step {state['best_step']}; adapter in {out / 'best'}")


if __name__ == "__main__":
    main()
