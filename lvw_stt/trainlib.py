"""Training loop shared by the Parakeet and Qwen trainers: cost-bucketed batches, AdamW + cosine, grad accumulation,
checkpoint/resume, best-on-dev selection, MPS memory logging, and a --smoke mode that checks all of it in minutes."""
from __future__ import annotations

import json
import math
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from lvw_stt.data import DATA, read_jsonl
from lvw_stt.disk import ROOT, free_gb, guard


def mem_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0


def device() -> torch.device:
    # NVIDIA (rented box) first, then Apple Metal, then CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_rows(codes: list[str], smoke: bool, seed: int) -> list[dict]:
    rows = [r for r in read_jsonl(DATA / "train" / "manifest.jsonl") if r["lang"] in codes]
    assert all(r["split"] == "train" for r in rows), "non-train row in the training manifest"
    if not rows:
        raise SystemExit("no training data; run: bash run.sh prepare")
    if smoke:
        random.Random(seed).shuffle(rows)
        rows = rows[:64]
    return rows


def dev_rows(codes: list[str], per_lang: int) -> list[dict]:
    rows, n = [], {}
    for r in read_jsonl(DATA / "eval" / "fleurs-dev.jsonl"):
        if r["lang"] in codes and n.get(r["lang"], 0) < per_lang:
            rows.append(r); n[r["lang"]] = n.get(r["lang"], 0) + 1
    return rows


def buckets(rows: list[dict], cost: Callable[[dict], float], budget: float, max_batch: int, seed: int) -> list[list[dict]]:
    """Sort by cost inside shuffled pools of 512, pack until the budget, shuffle batch order."""
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    out = []
    for i in range(0, len(rows), 512):
        pool = sorted(rows[i:i + 512], key=cost)
        b, mx = [], 0.0
        for r in pool:
            c = cost(r)
            if b and (max(mx, c) * (len(b) + 1) > budget or len(b) >= max_batch):
                out.append(b); b, mx = [], 0.0
            b.append(r); mx = max(mx, c)
        if b:
            out.append(b)
    rng.shuffle(out)
    return out


@dataclass
class Cfg:
    out: Path
    epochs: float = 2.0
    grad_acc: int = 4
    warmup: int = 300
    eval_every: int = 500
    save_every: int = 500
    keep: int = 2
    patience: int = 6
    clip: float = 1.0
    seed: int = 43
    smoke: bool = False
    resume: bool = False
    log_every: int = 10
    max_steps: int | None = None
    optim: str = "adamw"          # "adafactor": factored 2nd moment, ~no optimizer memory (prune heal)
    extra: dict = field(default_factory=dict)


def latest(out: Path) -> Path | None:
    c = sorted(out.glob("ckpt-*"), key=lambda p: int(p.name.split("-")[1]))
    return c[-1] if c else None


def loop(cfg: Cfg, model, param_groups: list[dict], batches_for_epoch: Callable[[int], list],
         step_fn: Callable[[list], torch.Tensor], eval_fn: Callable[[], float],
         save_fn: Callable[[Path], None], load_fn: Callable[[Path], None]) -> dict:
    """Optimizer steps count accumulated micro-batches. Returns summary (best dev score, steps, peak memory)."""
    cfg.out.mkdir(parents=True, exist_ok=True)
    log = open(cfg.out / "log.jsonl", "a")
    if cfg.optim == "adafactor":
        from transformers.optimization import Adafactor
        opt = Adafactor(param_groups, scale_parameter=False, relative_step=False, warmup_init=False,
                        weight_decay=0.0, clip_threshold=1.0)
    else:
        opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), weight_decay=0.01)
    n_micro = len(batches_for_epoch(0))
    total = cfg.max_steps or max(1, math.ceil(n_micro * cfg.epochs / cfg.grad_acc))
    warm = min(cfg.warmup, max(1, total // 10))

    def lr_mult(s):
        if s < warm:
            return (s + 1) / warm
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    state = {"step": 0, "micro": 0, "epoch": 0, "best": float("inf"), "best_step": -1, "bad": 0, "peak_gb": 0.0}
    if cfg.resume and (ck := latest(cfg.out)):
        load_fn(ck)
        tr = torch.load(ck / "trainer.pt", weights_only=False)
        opt.load_state_dict(tr["opt"]); sched.load_state_dict(tr["sched"]); state.update(tr["state"])
        random.setstate(tr["py_rng"]); torch.set_rng_state(tr["torch_rng"])
        print(f"resumed from {ck} at step {state['step']}")

    def checkpoint(tag: str):
        guard(1.0, what="checkpoint")
        d = cfg.out / tag
        tmp = d.with_name(d.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        save_fn(tmp)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "state": state,
                    "py_rng": random.getstate(), "torch_rng": torch.get_rng_state()}, tmp / "trainer.pt")
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)
        if tag.startswith("ckpt-"):
            for old in sorted(cfg.out.glob("ckpt-*"), key=lambda p: int(p.name.split("-")[1]))[:-cfg.keep]:
                shutil.rmtree(old)

    def evaluate():
        model.eval()
        with torch.no_grad():
            s = eval_fn()
        model.train()
        better = s < state["best"]
        if better:
            state.update(best=s, best_step=state["step"], bad=0)
            shutil.rmtree(cfg.out / "best", ignore_errors=True)
            save_fn(cfg.out / "best")
        else:
            state["bad"] += 1
        rec = {"step": state["step"], "dev": s, "best": state["best"], "t": time.time()}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[eval] step {state['step']} dev {s:.3f} (best {state['best']:.3f} @ {state['best_step']})")

    model.train()
    if state["step"] == 0 and not cfg.smoke:
        evaluate()                                  # step-0 dev score = the starting point
    t0, losses, done = time.time(), [], False
    while not done and state["epoch"] < math.ceil(cfg.epochs):
        batches = batches_for_epoch(state["epoch"])
        start = state["micro"] % len(batches) if state["micro"] else 0
        for bi in range(start, len(batches)):
            loss = step_fn(batches[bi])
            (loss / cfg.grad_acc).backward()
            losses.append(float(loss.detach()))
            state["micro"] += 1
            if state["micro"] % cfg.grad_acc:
                continue
            torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], cfg.clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            state["step"] += 1
            state["peak_gb"] = max(state["peak_gb"], mem_gb())
            if state["step"] % cfg.log_every == 0:
                rec = {"step": state["step"], "loss": sum(losses) / len(losses), "lr": sched.get_last_lr()[0],
                       "mem_gb": round(mem_gb(), 2), "free_disk_gb": round(free_gb(), 1),
                       "s_per_step": (time.time() - t0) / cfg.log_every}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"step {rec['step']:6d}/{total} loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
                      f"mem {rec['mem_gb']:.1f} GB  {rec['s_per_step']:.2f} s/step")
                losses, t0 = [], time.time()
            if not cfg.smoke and state["step"] % cfg.eval_every == 0:
                evaluate()
                if state["bad"] >= cfg.patience:
                    print("early stop: dev did not improve"); done = True
            if not cfg.smoke and state["step"] % cfg.save_every == 0:
                checkpoint(f"ckpt-{state['step']}")
            if state["step"] >= total:
                done = True
            if done:
                break
        else:
            state["epoch"] += 1; state["micro"] = 0
    if not cfg.smoke:
        evaluate()
        checkpoint(f"ckpt-{state['step']}")
    log.close()
    return state


def smoke_report(before: float, after: float, reloaded: float, peak_gb: float, mem_limit: float = 13.0,
                 rel_tol: float = 1e-3) -> bool:
    """Loss on a fixed batch must fall after 50 steps, a save/load round trip must reproduce it, memory must fit."""
    checks = {"loss falls": after < before, "reload matches": abs(reloaded - after) < rel_tol * max(1.0, abs(after)),
              f"peak < {mem_limit} GB": peak_gb < mem_limit}
    print(f"smoke: fixed-batch loss {before:.4f} -> {after:.4f} (reloaded {reloaded:.4f}); peak {peak_gb:.1f} GB")
    for k, v in checks.items():
        print(f"  {'ok  ' if v else 'FAIL'} {k}")
    ok = all(checks.values())
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return ok
