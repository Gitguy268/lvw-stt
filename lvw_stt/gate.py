"""Publish gates. Reads results/*.json, writes results/gate.json, exits non-zero if a checked gate fails.

  python -m lvw_stt.gate main    fine-tuned vs baseline: avg FLEURS-test error >= 3% better (relative),
                                 no language > 5% worse, CV22-test avg not worse
  python -m lvw_stt.gate fast    pruned vs fine-tuned: avg within 5% (relative) and >= 1.2x faster
"""
from __future__ import annotations

import json
import sys

from lvw_stt.paths import RESULTS

PAIRS = {"main": [("parakeet", "baseline", "phonon-base", "ft", "parakeet-ft"), ("qwen", "baseline", "qwen-base", "ft", "qwen-ft")],
         "fast": [("parakeet", "ft", "parakeet-ft", "fast", "parakeet-fast"), ("qwen", "ft", "qwen-ft", "fast", "qwen-fast")]}


def load(tag, model, s):
    for suffix in ("-full", ""):          # prefer full test sets when both exist
        p = RESULTS / f"{tag}-{model}-{s}{suffix}.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def check(kind: str) -> dict:
    out = {}
    for fam, t0, m0, t1, m1 in PAIRS[kind]:
        a, b = load(t0, m0, "fleurs-test"), load(t1, m1, "fleurs-test")
        if not (a and b):
            out[fam] = {"status": "missing results"}
            continue
        rel = (a["avg"] - b["avg"]) / a["avg"]
        common = sorted(set(a["per_lang"]) & set(b["per_lang"]))
        worse = {c: round(100 * (b["per_lang"][c]["err"] - a["per_lang"][c]["err"]) / max(a["per_lang"][c]["err"], 1e-6), 1)
                 for c in common}
        r = {"ref_avg": a["avg"], "new_avg": b["avg"], "rel_change_pct": round(-100 * rel, 2), "per_lang_rel_pct": worse,
             "set": b["set"]}
        if kind == "main":
            ca, cb = load(t0, m0, "cv22-test"), load(t1, m1, "cv22-test")
            r["cv22"] = (ca["avg"], cb["avg"]) if ca and cb else None
            fails = []
            if rel < 0.03:
                fails.append(f"avg improved {100 * rel:.2f}% < 3%")
            bad = {c: v for c, v in worse.items() if v > 5}
            if bad:
                fails.append(f"languages >5% worse: {bad}")
            if r["cv22"] and r["cv22"][1] > r["cv22"][0]:
                fails.append(f"CV22 avg worse {r['cv22'][0]} -> {r['cv22'][1]}")
        else:
            speed = b["rtf"] / max(a["rtf"], 1e-9)
            r["speedup"] = round(speed, 2)
            fails = []
            if -rel > 0.05:
                fails.append(f"avg {100 * -rel:.2f}% worse > 5%")
            if speed < 1.2:
                fails.append(f"speedup {speed:.2f}x < 1.2x")
        r["status"] = "PASS" if not fails else "FAIL"
        r["fails"] = fails
        out[fam] = r
    return out


def main() -> None:
    kind = sys.argv[1] if len(sys.argv) > 1 else "main"
    res = check(kind)
    g = json.loads((RESULTS / "gate.json").read_text()) if (RESULTS / "gate.json").exists() else {}
    g[kind] = res
    (RESULTS / "gate.json").write_text(json.dumps(g, indent=1))
    for fam, r in res.items():
        line = f"{kind:4s} {fam:8s} {r['status']}"
        if "new_avg" in r:
            line += f"  {r['ref_avg']:.2f} -> {r['new_avg']:.2f} ({r['rel_change_pct']:+.2f}%)"
        print(line, "; ".join(r.get("fails", [])))
    if any(r["status"] == "FAIL" for r in res.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
