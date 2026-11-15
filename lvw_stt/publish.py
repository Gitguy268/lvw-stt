"""Publish to the Hub. Explicit only, and private first.

  python -m lvw_stt.publish                 # gated models -> private repos (safetensors + GGUF ladders), cards, eval dataset
  python -m lvw_stt.publish --cards-only    # regenerate and upload model cards
  python -m lvw_stt.publish --public        # after you have looked at the private repos: make them public + collection

A model is only uploaded when results/gate.json says PASS for it (main gate; Fast gate for -Fast).
With LVW_PROFILE=en-nl the only target is LvW-STT-EN-NL-0.6B (gate en-nl, results/en-nl/gate.json).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

from lvw_stt import langs
from lvw_stt.data import DATA, read_jsonl
from lvw_stt.disk import ROOT
from lvw_stt.paths import (EN_NL_DEV, EN_NL_NAME, EN_NL_SETS, ESB_SETS, EU_NAME, HF_USER, PROFILE, QW_NAME, RELEASE,
                           RESULTS, TRAIN_DIR)

EVAL_REPO = f"{HF_USER}/LvW-STT-EN-NL-eval" if PROFILE == "en-nl" else f"{HF_USER}/LvW-STT-eval"
DATASETS = ["google/fleurs", "fsicoli/common_voice_22_0", "facebook/voxpopuli"]


def res(tag, model, s):
    for suf in ("-full", ""):
        p = RESULTS / f"{tag}-{model}-{s}{suf}.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def gate_status(kind: str, fam: str) -> str:
    p = RESULTS / "gate.json"
    return json.loads(p.read_text()).get(kind, {}).get(fam, {}).get("status", "missing") if p.exists() else "missing"


def wer_table(base, new, base_name, new_name) -> str:
    rows = [f"| Language | {base_name} | {new_name} |", "|---|--:|--:|"]
    for c in sorted(new["per_lang"], key=lambda c: langs.TABLE[c][0]):
        b, n = base["per_lang"].get(c, {}).get("err"), new["per_lang"][c]["err"]
        bs = f"{b:.2f}" if b is not None else "–"
        ns = f"**{n:.2f}**" if b is not None and n < b else f"{n:.2f}"
        rows.append(f"| {c}{' (CER)' if c in langs.CER_LANGS else ''} | {bs} | {ns} |")
    rows.append(f"| **Average of {new['n_langs']}** | {base['avg']:.2f} | **{new['avg']:.2f}** |")
    return "\n".join(rows)


def gguf_table(family: str, name: str) -> str:
    note = ("100 utterances per language (English, Dutch) from FLEURS test" if family == "en-nl"
            else "50 utterances per language from FLEURS test")
    p = RESULTS / f"gguf-{family}.json"
    if not p.exists():
        return "_GGUF results pending._"
    rows = json.loads(p.read_text())
    out = ["| File | Size | FLEURS avg error* | vs top precision | Speed (x real time, M4) |", "|---|--:|--:|--:|--:|"]
    for r in sorted(rows, key=lambda r: -r["bytes"]):
        flag = " (not recommended)" if r["rel_vs_top_pct"] > 10 else ""
        out.append(f"| [{r['file']}](https://huggingface.co/{HF_USER}/{name}-GGUF/resolve/main/{r['file']}) | "
                   f"{r['bytes'] / 1e6:,.0f} MB | {r['avg']:.2f} | {r['rel_vs_top_pct']:+.1f}%{flag} | {r['rtf']:.0f} |")
    out.append(f"\n\\* {note}, same scoring as the main table.")
    return "\n".join(out)


def data_hours(codes) -> str:
    rows = [r for r in read_jsonl(TRAIN_DIR / "manifest.jsonl") if r["lang"] in codes]
    by = {}
    for r in rows:
        by[r["source"]] = by.get(r["source"], 0) + r["dur"] / 3600
    return ", ".join(f"{k} {v:.0f} h" for k, v in sorted(by.items())) + f" (total {sum(by.values()):.0f} h)"


def card_eu(fast: bool) -> str:
    name = f"{EU_NAME}-Fast" if fast else EU_NAME
    base = res("ft", "parakeet-ft", "fleurs-test") if fast else res("baseline", "phonon-base", "fleurs-test")
    new = res("fast" if fast else "ft", "parakeet-fast" if fast else "parakeet-ft", "fleurs-test")
    cb = res("baseline", "phonon-base", "cv22-test"); cn = res("fast" if fast else "ft", "parakeet-fast" if fast else "parakeet-ft", "cv22-test")
    base_name = EU_NAME if fast else "Phonon-2.5"
    return f"""---
license: cc-by-4.0
base_model: {"Lucasgitguy-co/" + EU_NAME if fast else "FermionResearch/Phonon-2.5"}
base_model_relation: finetune
library_name: transformers
pipeline_tag: automatic-speech-recognition
language: [{", ".join(langs.PHONON_LANGS)}]
datasets: [{", ".join(DATASETS)}]
tags: [speech-to-text, asr, parakeet, tdt, multilingual, gguf]
---

# {name}

Speech recognition for 25 European languages. {"Pruned" if fast else "Fine-tuned"} from {base_name}{" (24 → 18 encoder layers) for speed" if fast else ", the compressed Parakeet TDT 0.6B v3 student by Fermion Research"}. GGUF files in every size: [{name}-GGUF](https://huggingface.co/{HF_USER}/{name}-GGUF).

## Benchmarks

Word error rate, %, lower is better ({new['set']}; Whisper-style normalisation, average over languages).

{wer_table(base, new, base_name, name)}

Common Voice 22 test (out of domain for FLEURS): {cb['avg'] if cb and not fast else '–'} → {cn['avg'] if cn else '–'}.

## Run it

```bash
# transformers
python -c "from transformers import pipeline; print(pipeline('automatic-speech-recognition', '{HF_USER}/{name}')('audio.wav'))"
# transcribe.cpp (GGUF, Metal/CPU/Vulkan)
transcribe-cli -m {name}-Q8_0.gguf audio.wav
```

<details><summary>Training</summary>

- Data: {data_hours(langs.PHONON_LANGS)} from FLEURS train (CC-BY-4.0), Common Voice 22 train (CC0) and VoxPopuli train (CC0). Cased, punctuated transcripts only. No test or dev audio was trained on. FLEURS *train* is in the mix, so the Common Voice line above is the out-of-domain check.
- Method: {"encoder layers chosen by block influence were removed, then the rest was fully fine-tuned (Adafactor) with TDT loss plus encoder-feature distillation from " + EU_NAME if fast else "Phonon-2.5's five-value container was expanded to dense fp32, then LoRA (r=32) on every encoder attention and feed-forward projection, merged, plus a full update of the LSTM decoder, joint and encoder projector, trained with transformers' TDT loss and SpecAugment"}.
- Hardware: one Apple M4 (16 GB), PyTorch MPS.
- Code: the `lvw-stt` scripts (setup, data, train, evaluate, GGUF, publish).
</details>

<details><summary>Licence and credits</summary>

CC-BY-4.0. Derived from [Phonon-2.5](https://huggingface.co/FermionResearch/Phonon-2.5) by Fermion Research and [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) by NVIDIA; see `NOTICE`. Not an official Fermion Research or NVIDIA release.
</details>
"""


def card_qw(fast: bool) -> str:
    name = f"{QW_NAME}-Fast" if fast else QW_NAME
    base = res("ft", "qwen-ft", "fleurs-test") if fast else res("baseline", "qwen-base", "fleurs-test")
    new = res("fast" if fast else "ft", "qwen-fast" if fast else "qwen-ft", "fleurs-test")
    base_name = QW_NAME if fast else "Qwen3-ASR-0.6B"
    return f"""---
license: apache-2.0
base_model: {"Lucasgitguy-co/" + QW_NAME if fast else "Qwen/Qwen3-ASR-0.6B"}
base_model_relation: finetune
pipeline_tag: automatic-speech-recognition
language: [{", ".join(langs.QWEN_LANGS)}]
datasets: [{", ".join(DATASETS)}]
tags: [speech-to-text, asr, qwen3-asr, multilingual, gguf]
---

# {name}

Speech recognition and language identification for 30 languages. {"Decoder pruned (28 → 20 layers) from " + QW_NAME + " for speed" if fast else "LoRA fine-tune of Qwen3-ASR-0.6B, merged"}. GGUF: [{name}-GGUF](https://huggingface.co/{HF_USER}/{name}-GGUF).

## Benchmarks

Error rate, %, lower is better ({new['set']}; WER, CER where marked; no language hint given, as in Qwen's own evaluation).

{wer_table(base, new, base_name, name)}

## Run it

```bash
pip install -U qwen-asr
python -c "from qwen_asr import Qwen3ASRModel; m=Qwen3ASRModel.from_pretrained('{HF_USER}/{name}'); print(m.transcribe(audio='audio.wav')[0].text)"
# llama.cpp
llama-mtmd-cli -m {name}-Q8_0.gguf --mmproj mmproj-{name}-F16.gguf --audio audio.wav -p "Transcribe."
```

<details><summary>Training</summary>

- Data: {data_hours(langs.QWEN_LANGS)}; FLEURS train (CC-BY-4.0), Common Voice 22 train (CC0), VoxPopuli train (CC0). No test or dev audio was trained on.
- Method: {"contiguous decoder block with the smallest angular distance removed, remaining decoder layers fully fine-tuned (Adafactor) with cross-entropy plus KL distillation from " + QW_NAME if fast else "LoRA r=32 on decoder attention/MLP and audio-encoder attention, prompt layout from QwenLM's finetuning script (`language X<asr_text>...`)"}.
- Hardware: one Apple M4 (16 GB), PyTorch MPS, bf16.
</details>

<details><summary>Licence and credits</summary>

Apache-2.0. Derived from [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) by the Qwen team; see `NOTICE`. Not an official Qwen release.
</details>
"""


SET_NAMES = {"esb-ls-clean": "LibriSpeech test-clean", "esb-ls-other": "LibriSpeech test-other", "esb-ami": "AMI",
             "esb-earnings22": "Earnings-22", "esb-gigaspeech": "GigaSpeech", "esb-spgispeech": "SPGISpeech",
             "esb-voxpopuli": "VoxPopuli", "fleurs-test-en-nl": "FLEURS test", "cv22-test-en-nl": "Common Voice 22 test",
             "voxpopuli-test-nl": "VoxPopuli test"}


def en_nl_table(tag: str, model: str) -> str:
    """WER per set: stock Phonon-2.5-English, stock Phonon-2.5 (25 languages), this model."""
    cols = [("baseline", "phonon-en-base"), ("baseline", "phonon-multi"), (tag, model)]

    def e(t, m, s, lang):
        r = res(t, m, s)
        return r["per_lang"][lang]["err"] if r and lang in r["per_lang"] else None

    def n(s, lang):
        r = res(tag, model, s)
        return r["per_lang"][lang]["n"] if r and lang in r["per_lang"] else 0

    out = ["| Language | Test set (clips) | Phonon-2.5 English | Phonon-2.5 (25 languages) | " + EN_NL_NAME + " |",
           "|---|---|--:|--:|--:|"]

    def row(lang, label, vals):
        best = min(v for v in vals if v is not None) if any(v is not None for v in vals) else None
        cells = ["–" if v is None else (f"**{v:.2f}**" if v == best else f"{v:.2f}") for v in vals]
        out.append(f"| {lang} | {label} | " + " | ".join(cells) + " |")

    for s in ESB_SETS + ["fleurs-test-en-nl", "cv22-test-en-nl"]:
        row("English", f"{SET_NAMES[s]} ({n(s, 'en')})", [e(t, m, s, "en") for t, m in cols])
    avg = []
    for t, m in cols:
        v = [e(t, m, s, "en") for s in ESB_SETS]
        avg.append(sum(v) / len(v) if all(x is not None for x in v) else None)
    row("English", "**Average of the 7 leaderboard sets**", avg)
    nl_sets = ["fleurs-test-en-nl", "cv22-test-en-nl", "voxpopuli-test-nl"]
    for s in nl_sets:
        row("Dutch", f"{SET_NAMES[s]} ({n(s, 'nl')})", [e(t, m, s, "nl") for t, m in cols])
    avg = []
    for t, m in cols:
        v = [e(t, m, s, "nl") for s in nl_sets]
        avg.append(sum(v) / len(v) if all(x is not None for x in v) else None)
    row("Dutch", "**Average**", avg)
    return "\n".join(out)


def card_en_nl(tag: str = "ft", model: str = "en-nl-ft") -> str:
    name = EN_NL_NAME
    return f"""---
license: cc-by-4.0
base_model: FermionResearch/Phonon-2.5-English
base_model_relation: finetune
library_name: transformers
pipeline_tag: automatic-speech-recognition
language: [en, nl]
datasets: [{", ".join(DATASETS)}]
metrics: [wer]
tags: [speech-to-text, asr, parakeet, tdt, english, dutch, gguf]
---

# {name}

Speech recognition for English and Dutch. It is a fine-tune of [Phonon-2.5 English](https://huggingface.co/FermionResearch/Phonon-2.5-English), Fermion Research's compressed Parakeet TDT 0.6B v3 student, which was trained for English only. This version adds Dutch, and keeps English level with the original: the average over the 7 Open ASR Leaderboard sets is not worse, and no English set below is more than 3% worse. GGUF files in every size: [{name}-GGUF](https://huggingface.co/{HF_USER}/{name}-GGUF).

## Benchmarks

Word error rate, %, lower is better; the best value in each row is in bold. The scoring code and Whisper-style normalisation are the Open ASR Leaderboard's. The English sets are **random subsets** of the leaderboard's test sets (clip counts in brackets), not the full sets, so they are not directly comparable to leaderboard rows. All three models were scored on exactly the same clips.

{en_nl_table(tag, model)}

## Run it

```python
# pip install "transformers>=5.22" torch soundfile
import soundfile as sf, torch
from transformers import AutoProcessor, ParakeetForTDT
proc = AutoProcessor.from_pretrained("{HF_USER}/{name}")
model = ParakeetForTDT.from_pretrained("{HF_USER}/{name}").eval()
audio, sr = sf.read("audio.wav", dtype="float32")          # 16 kHz mono
inputs = proc([audio], sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    out = model.generate(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"])
print(proc.batch_decode(getattr(out, "sequences", out), skip_special_tokens=True)[0])
```

GGUF with [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (Metal, CPU, Vulkan): see [{name}-GGUF](https://huggingface.co/{HF_USER}/{name}-GGUF).

<details><summary>Training</summary>

- Data: {data_hours(["en", "nl"])}, English and Dutch from FLEURS train (CC-BY-4.0), Common Voice 22 train (CC0) and VoxPopuli train (CC0). Cased, punctuated transcripts only. About 60% Dutch, plus 40% English replay, so the model doesn't forget English. No test or dev audio was trained on. VoxPopuli and FLEURS train are in the mix, so their test sets are in domain; LibriSpeech, AMI, Earnings-22, GigaSpeech and SPGISpeech are out of domain.
- Method: Phonon-2.5 English's five-value container was expanded to dense fp32. Then LoRA (r=32) was trained on every encoder attention and feed-forward projection and merged, with a full update of the LSTM decoder, joint and encoder projector. Training used transformers' TDT loss and SpecAugment. The checkpoint was picked on an English + Dutch dev set (FLEURS, Common Voice and VoxPopuli dev).
- Release gate: published only because the English average over the 7 leaderboard sets is not worse than Phonon-2.5 English, no English set is more than 3% worse, and Dutch is clearly better than Phonon-2.5 English and close to Phonon-2.5.
- Hardware: one Apple M4 MacBook Air (16 GB), PyTorch MPS.
- Weights are dense fp32 (2.5 GB). For a small download use the GGUF files.
</details>

<details><summary>Licence and credits</summary>

CC-BY-4.0. Derived from [Phonon-2.5 English](https://huggingface.co/FermionResearch/Phonon-2.5-English) by Fermion Research and [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) by NVIDIA; see `NOTICE`, which also carries Fermion's notice (its training data includes CHiME-6 and SPGISpeech). Not an official Fermion Research or NVIDIA release.
</details>
"""


def card_gguf(name: str, family: str, runtime: str) -> str:
    lic = "apache-2.0" if family.startswith("qwen") else "cc-by-4.0"
    return f"""---
license: {lic}
base_model: {HF_USER}/{name}
base_model_relation: quantized
pipeline_tag: automatic-speech-recognition
tags: [gguf, {runtime.split('.')[0]}, speech-to-text]
---

# {name}-GGUF

GGUF files of [{name}](https://huggingface.co/{HF_USER}/{name}) for {runtime}.

{gguf_table(family, name)}

```bash
{"transcribe-cli -m " + name + "-Q8_0.gguf audio.wav" if not family.startswith("qwen") else "llama-server -m " + name + "-Q8_0.gguf --mmproj mmproj-" + name + "-F16.gguf  # then POST audio to /v1/chat/completions"}
```

Licence and credits as in the [main repo](https://huggingface.co/{HF_USER}/{name}).
"""


TARGETS_EN_NL = [("en-nl", "en-nl", "en-nl", EN_NL_NAME, lambda: card_en_nl(), "transcribe.cpp")]
TARGETS = [  # (family, kind, fam-key in gate, release name, card fn, runtime)
    ("eu", "main", "parakeet", EU_NAME, lambda: card_eu(False), "transcribe.cpp"),
    ("qwen", "main", "qwen", QW_NAME, lambda: card_qw(False), "llama.cpp"),
    ("eu-fast", "fast", "parakeet", f"{EU_NAME}-Fast", lambda: card_eu(True), "transcribe.cpp"),
    ("qwen-fast", "fast", "qwen", f"{QW_NAME}-Fast", lambda: card_qw(True), "llama.cpp"),
]


def verify_upload(api: HfApi, repo: str, d: Path) -> None:
    """Every local file must be in the repo with the same size (LFS files also by sha256)."""
    import hashlib
    remote = {f.path: f for f in api.get_paths_info(repo, [p.name for p in d.iterdir() if p.is_file()])}
    for p in d.iterdir():
        if not p.is_file():
            continue
        r = remote.get(p.name)
        if r is None or r.size != p.stat().st_size:
            raise SystemExit(f"upload check failed for {repo}/{p.name}")
        if getattr(r, "lfs", None):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
            if h.hexdigest() != r.lfs.sha256:
                raise SystemExit(f"upload checksum differs for {repo}/{p.name}")
    print(f"upload verified: {repo} ({len(remote)} files)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards-only", action="store_true")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--only", nargs="*", help="families: eu qwen eu-fast qwen-fast")
    a = ap.parse_args()
    api = HfApi()
    done = []
    for family, kind, gk, name, card, runtime in (TARGETS_EN_NL if PROFILE == "en-nl" else TARGETS):
        if a.only and family not in a.only:
            continue
        repo = f"{HF_USER}/{name}"
        st = gate_status(kind, gk)
        if st != "PASS":
            print(f"{name}: gate {kind}/{gk} = {st}; not published"); continue
        if a.public:
            for r in (repo, repo + "-GGUF"):
                api.update_repo_settings(r, private=False); print("public:", r)
            done.append(repo); continue
        api.create_repo(repo, private=True, exist_ok=True)
        d = RELEASE / name
        (d / "README.md").write_text(card())
        if a.cards_only:
            api.upload_file(path_or_fileobj=str(d / "README.md"), path_in_repo="README.md", repo_id=repo)
        else:
            api.upload_folder(folder_path=str(d), repo_id=repo, commit_message=f"{name}")
            verify_upload(api, repo, d)
            py = "parakeet" if PROFILE == "en-nl" else "qwen"
            subprocess.run([str(ROOT / "envs" / py / "bin" / "python"), "-m", "lvw_stt.gguf.ladder",
                            "--family", family, "--upload"], cwd=ROOT, check=True)
        gg = RESULTS / f"README-{name}-GGUF.md"
        gg.write_text(card_gguf(name, family, runtime))
        api.create_repo(repo + "-GGUF", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(gg), path_in_repo="README.md", repo_id=repo + "-GGUF")
        done.append(repo)
        print("uploaded (private):", repo, repo + "-GGUF")
    if not a.public and not a.cards_only and done:
        api.create_repo(EVAL_REPO, repo_type="dataset", private=True, exist_ok=True)
        tmp = RESULTS / "eval-repo"
        tmp.mkdir(exist_ok=True)
        keep = ("id", "lang", "source", "split", "text", "dur")
        sets = EN_NL_SETS + [EN_NL_DEV.stem] if PROFILE == "en-nl" else ["fleurs-test", "fleurs-dev", "cv22-test"]
        for src, dst in [(TRAIN_DIR / "manifest.jsonl", "train-manifest.jsonl")] + \
                [(DATA / "eval" / f"{s}.jsonl", f"{s}-manifest.jsonl") for s in sets]:
            with open(tmp / dst, "w", encoding="utf-8") as f:
                for r in read_jsonl(src):
                    f.write(json.dumps({k: r[k] for k in keep}, ensure_ascii=False) + "\n")
        api.upload_folder(folder_path=str(tmp), repo_id=EVAL_REPO, repo_type="dataset", path_in_repo="manifests")
        api.upload_folder(folder_path=str(RESULTS), repo_id=EVAL_REPO, repo_type="dataset", path_in_repo="results",
                          allow_patterns=["*.json", "*.md"])
        print("uploaded (private):", EVAL_REPO)
    if a.public and done:
        api.update_repo_settings(EVAL_REPO, private=False, repo_type="dataset")
        c = api.create_collection(title="LvW-STT", namespace=HF_USER, private=False, exists_ok=True,
                                  description="Speech-to-text fine-tunes of Phonon and Qwen3-ASR models, with GGUF ladders")
        for r in done:
            for item in (r, r + "-GGUF"):
                api.add_collection_item(c.slug, item_id=item, item_type="model", exists_ok=True)
        api.add_collection_item(c.slug, item_id=EVAL_REPO, item_type="dataset", exists_ok=True)
        print("collection:", c.url)


if __name__ == "__main__":
    main()
