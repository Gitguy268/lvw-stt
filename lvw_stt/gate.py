"""Publish gates. Reads results/*.json, writes results/gate.json, exits non-zero if a checked gate fails.

  python -m lvw_stt.gate main    fine-tuned vs baseline: avg FLEURS-test error >= 3% better (relative),
                                 no language > 5% worse, CV22-test avg not worse
  python -m lvw_stt.gate fast    pruned vs fine-tuned: avg within 5% (relative) and >= 1.2x faster
  LVW_PROFILE=en-nl python -m lvw_stt.gate calibrate
                                 stock Phonon-2.5-English on our ESB subsets must match its model card
  LVW_PROFILE=en-nl python -m lvw_stt.gate en-nl
                                 English: ESB-7 average not worse, no English set > 3% worse (relative);
                                 Dutch: >= 3% better than stock Phonon-2.5-English, <= 5% worse than Phonon-2.5
"""
from __future__ import annotations

import json
import sys

from lvw_stt.paths import ESB_SETS, PROFILE, RESULTS

# Phonon-2.5-English model card (Open ASR Leaderboard code, full test sets), WER %
CARD_EN = {"esb-ls-clean": 1.64, "esb-ls-other": 3.56, "esb-ami": 8.76, "esb-earnings22": 6.22,
           "esb-gigaspeech": 7.99, "esb-spgispeech": 3.31, "esb-voxpopuli": 2.28}
EN_SETS = ESB_SETS + ["fleurs-test-en-nl", "cv22-test-en-nl"]       # each must not get > 3% worse
NL_SETS = ["fleurs-test-en-nl", "cv22-test-en-nl", "voxpopuli-test-nl"]

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


def err(tag: str, model: str, s: str, lang: str) -> float:
    r = load(tag, model, s)
    if not r or lang not in r["per_lang"]:
        raise SystemExit(f"missing result: {tag}-{model}-{s}.json ({lang}); run the evaluate step first")
    return r["per_lang"][lang]["err"]


def calibrate() -> dict:
    """If our scoring of the stock model is far from its card, the expansion or the scoring is wrong: stop."""
    rows, fails = {}, []
    for s, card in CARD_EN.items():
        ours = err("baseline", "phonon-en-base", s, "en")
        tol = max(0.6, 0.30 * card)
        ok = abs(ours - card) <= tol
        rows[s] = {"card": card, "ours": ours, "tolerance": round(tol, 2), "ok": ok}
        print(f"  {s:16s} card {card:5.2f}  ours {ours:5.2f}  {'ok' if ok else 'FAIL'} (±{tol:.2f})")
        if not ok:
            fails.append(f"{s}: {ours:.2f} vs card {card:.2f}")
    return {"calibrate": {"status": "PASS" if not fails else "FAIL", "fails": fails, "sets": rows}}


def en_nl() -> dict:
    fails, en_rows = [], {}
    for s in EN_SETS:
        b, n = err("baseline", "phonon-en-base", s, "en"), err("ft", "en-nl-ft", s, "en")
        rel = 100 * (n - b) / max(b, 1e-6)
        en_rows[s] = {"base": b, "new": n, "rel_pct": round(rel, 2)}
        if rel > 3.0:
            fails.append(f"English {s} {rel:+.2f}% worse (> 3%)")
    esb_b = sum(en_rows[s]["base"] for s in ESB_SETS) / len(ESB_SETS)
    esb_n = sum(en_rows[s]["new"] for s in ESB_SETS) / len(ESB_SETS)
    if esb_n > esb_b:
        fails.append(f"English ESB-7 average worse: {esb_b:.3f} -> {esb_n:.3f}")
    nl = {m: sum(err(t, m, s, "nl") for s in NL_SETS) / len(NL_SETS)
          for t, m in (("baseline", "phonon-en-base"), ("baseline", "phonon-multi"), ("ft", "en-nl-ft"))}
    if nl["en-nl-ft"] > 0.97 * nl["phonon-en-base"]:
        fails.append(f"Dutch not >= 3% better than Phonon-2.5-English: {nl['phonon-en-base']:.2f} -> {nl['en-nl-ft']:.2f}")
    if nl["en-nl-ft"] > 1.05 * nl["phonon-multi"]:
        fails.append(f"Dutch > 5% worse than Phonon-2.5 (25 languages): {nl['phonon-multi']:.2f} vs {nl['en-nl-ft']:.2f}")
    r = {"status": "PASS" if not fails else "FAIL", "fails": fails, "english": en_rows,
         "esb_avg": {"base": round(esb_b, 3), "new": round(esb_n, 3)}, "dutch_avg": {k: round(v, 3) for k, v in nl.items()}}
    for s, v in en_rows.items():
        print(f"  en {s:18s} {v['base']:6.2f} -> {v['new']:6.2f} ({v['rel_pct']:+.2f}%)")
    print(f"  en ESB-7 average     {esb_b:6.2f} -> {esb_n:6.2f}")
    print(f"  nl average           Phonon-2.5-English {nl['phonon-en-base']:.2f}, Phonon-2.5 {nl['phonon-multi']:.2f}, "
          f"ours {nl['en-nl-ft']:.2f}")
    return {"en-nl": r}


def main() -> None:
    kind = sys.argv[1] if len(sys.argv) > 1 else "main"
    if kind in ("calibrate", "en-nl"):
        if PROFILE != "en-nl":
            raise SystemExit("run with LVW_PROFILE=en-nl")
        res = calibrate() if kind == "calibrate" else en_nl()
    elif PROFILE == "en-nl":
        raise SystemExit(f"gate {kind} belongs to the eu profile")
    else:
        res = check(kind)
    RESULTS.mkdir(parents=True, exist_ok=True)
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
