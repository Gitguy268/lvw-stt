"""Dry run of the stages that have never run (export -> GGUF -> parity -> quantize -> cards), on the STOCK
Phonon-2.5-English, before any training. Nothing is uploaded.

  LVW_PROFILE=en-nl python -m lvw_stt.dryrun

The release is written in fp16 here (1.25 GB instead of 2.5 GB) to fit the disk; transcribe.cpp upcasts F16 to F32
losslessly, and the HF reference loads the same file, so parity is still exact. It is deleted after the parity check.
"""
from __future__ import annotations

import argparse
import json
import re

import torch

from lvw_stt.disk import free_gb
from lvw_stt.paths import BASE_DENSE, EN_NL_NAME, PROFILE, PROFILE_RUNS, RESULTS, TRANSCRIBE_CLI, TRANSCRIBE_QUANT


def main() -> None:
    if PROFILE != "en-nl":
        raise SystemExit("run with LVW_PROFILE=en-nl")
    for b in (TRANSCRIBE_CLI, TRANSCRIBE_QUANT):
        if not b.exists():
            raise SystemExit(f"{b} missing: run `bash run.sh setup` (eu profile) first")
    if not (RESULTS / "baseline-phonon-en-base-esb-ls-clean.json").exists():
        raise SystemExit("no baseline results yet: run the calibrate step first")

    from lvw_stt.gguf import ladder
    from lvw_stt.parakeet.expand_phonon import load_parakeet
    from lvw_stt.parakeet.export import write_release
    from lvw_stt.publish import card_en_nl, card_gguf

    out = PROFILE_RUNS / "dryrun" / EN_NL_NAME
    print(f"1/4 write the stock model as a release ({free_gb():.1f} GB free)")
    if not (out / "model.safetensors").exists():
        write_release(load_parakeet(BASE_DENSE).half(), out, EN_NL_NAME, extra="(dry run: stock weights)")
    for f in ("config.json", "processor_config.json", "tokenizer.json", "generation_config.json", "LICENSE", "NOTICE"):
        if not (out / f).exists():
            raise SystemExit(f"release is missing {f}")
    if "UPSTREAM NOTICE" not in (out / "NOTICE").read_text() or "CHiME-6" not in (out / "NOTICE").read_text():
        raise SystemExit("NOTICE does not carry Fermion's upstream notice")

    print("2/4 GGUF: convert -> parity vs HF -> Q8_0 + Q4_K_M (nothing uploaded)")
    a = argparse.Namespace(quant=["Q8_0", "Q4_K_M"], upload=False, keep_top=False)
    ladder.ladder_parakeet(a, EN_NL_NAME, "en-nl-dryrun", "en-nl-dryrun", out,
                           hf_args=["--model", "en-nl-ft", "--path", str(out)], drop_src_after_parity=True)
    rows = {r["quant"]: r for r in json.loads((RESULTS / "gguf-en-nl-dryrun.json").read_text())}
    for q in ("Q8_0", "Q4_K_M"):
        rel = rows[q]["rel_vs_top_pct"]
        print(f"  {q}: {rows[q]['bytes'] / 1e6:.0f} MB, error {rows[q]['avg']:.2f} ({rel:+.1f}% vs F32)")
        if rel > 25:
            raise SystemExit(f"{q} is {rel:.1f}% worse than F32: quantization is broken")

    print("3/4 cards: render and parse (baseline numbers stand in for the fine-tuned model)")
    from huggingface_hub import ModelCard
    checks = []
    for text, want in ((card_en_nl("baseline", "phonon-en-base"), {"base_model": "FermionResearch/Phonon-2.5-English"}),
                       (card_gguf(EN_NL_NAME, "en-nl-dryrun", "transcribe.cpp"), {"base_model": f"Lucasgitguy-co/{EN_NL_NAME}"})):
        data = ModelCard(text).data.to_dict()
        if data.get("license") != "cc-by-4.0" or any(data.get(k) != v for k, v in want.items()):
            raise SystemExit(f"card metadata wrong: {data}")
        if re.search(r"!\[", text):
            raise SystemExit("card contains an image (the cards are text only)")
        checks.append(data)
    table = card_en_nl("baseline", "phonon-en-base").split("## Benchmarks")[1].split("## Run it")[0]
    if "| – " in table or "| – |" in table:
        raise SystemExit("benchmark table has missing numbers:\n" + table)
    (PROFILE_RUNS / "dryrun").mkdir(parents=True, exist_ok=True)
    (PROFILE_RUNS / "dryrun" / "README-preview.md").write_text(card_en_nl("baseline", "phonon-en-base"))

    print("4/4 summary")
    summary = {"parity": rows["F32"]["parity"], "quants": {q: rows[q] for q in rows}, "card_metadata": checks[0],
               "dtype_note": "dry-run release was fp16; the real release is fp32", "torch": torch.__version__}
    (RESULTS / "dryrun.json").write_text(json.dumps(summary, indent=1))
    print(f"DRY RUN PASS: parity {100 * rows['F32']['parity']:.1f}%, card preview in "
          f"{PROFILE_RUNS / 'dryrun' / 'README-preview.md'}")


if __name__ == "__main__":
    main()
