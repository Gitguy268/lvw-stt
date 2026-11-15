# LvW-STT progress

Goal: fine-tune FermionResearch/Phonon-2.5 (Parakeet TDT, 25 EU langs) and Qwen/Qwen3-ASR-0.6B (30 langs) on the
local M4 for accuracy, ship GGUF ladders, and add gated pruned "Fast" variants. Publish to `Lucasgitguy-co`.
Plan: ~/.claude/plans/post-train-improve-typed-rivest.md

## Status (2026-11-10)

- [x] Code written: run.sh stages, data streaming, eval harness, both trainers, export, gates, prune+heal, GGUF ladder, publish
- [x] Syntax check (ast) of every module
- [x] `bash run.sh all`: the whole pipeline in one resumable command (steps marked in runs/.done/, log in runs/all-*.log)
- [x] setup, prepare-eval, expand, smoke-parakeet, smoke-qwen: done and verified (2026-11-10), marked in runs/.done/
- [ ] baseline, prepare, training, export, prune, GGUF, publish: start with `bash run.sh all` (it skips the steps above)

## Bugs found by the smoke tests (fixed)

- transcribe.cpp builds its quantizer only with -DTRANSCRIBE_BUILD_TOOLS=ON.
- Phonon's .tar.zst has its files at the top level and overwrote the repo's config.json; it now unpacks to unpacked/.
- Parakeet config has layerdrop 0.1: random layer skipping in train mode gave losses of ~2000. Now off.
- Parakeet BatchNorm used tiny-batch statistics in train mode. It is now frozen on stored statistics.
- Qwen prompt lengths measured as one padded batch were wrong (loss 7.7 vs 0.5). Now measured per clip.
- Qwen batch of 160 s audio swapped at ~380 s/step. It is now 40 s (max 4 clips) with grad-acc 16, at 2.4–4.4 s/step and 9.7 GB.
- PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 removed so an oversized batch fails fast instead of swapping.

Measured: Parakeet 1.2–2.8 s/step at 11.3 GB peak; Qwen 2.4–4.4 s/step at 9.7 GB peak; Phonon expansion 7.9% WER (en).
Export, prune, GGUF and publish have still never run; they come after training and are resumable.

## Pinned sources

| What | Revision |
|---|---|
| FermionResearch/Phonon-2.5 | 2af4bb8ccceb66a8ce06c4081bb968f1802c1e58 |
| Qwen/Qwen3-ASR-0.6B | 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 |
| nvidia/parakeet-tdt-0.6b-v3 (config/tokenizer only) | 541d1f99c6b0c3cd0b11a95167540bb8edefd82b |
| handy-computer/transcribe.cpp | 5700f127e38dd53c4735ea23ef743b4b538eacd4 |
| ggml-org/llama.cpp | a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5 |
| QwenLM/Qwen3-ASR | 7c6daf77a2421100f5fb066495372c00129d39ff |

## Decisions

- Two venvs: qwen-asr pins transformers 4.57.6; Parakeet TDT training needs transformers 5.22 (pure-PyTorch TDT loss).
- Dense Phonon is never written to disk (2.5 GB). `load_parakeet(runs/phonon25-dense)` rebuilds it from the 178 MB container.
- Training audio is 32 kbps Opus (about 15 MB/h, so about 330 h in 5 GB). Eval audio is FLAC. Cased, punctuated sources only (MLS and LibriSpeech were skipped).
- Gates: main means ≥3% better on average, no language more than 5% worse, CV22 not worse. Fast means within 5% and ≥1.2× faster. GGUF parity must be ≥98% (Parakeet F32) or ≥90% (Qwen BF16).
- Names: LvW-STT-EU-0.6B (Phonon-derived, CC-BY-4.0) and LvW-STT-Qwen3-0.6B (Apache-2.0), plus -Fast and -GGUF repos and the LvW-STT-eval dataset.
- No images on the model cards (user's choice).

## Untested assumptions to watch in the smoke runs

- `HfFileSystem` range reads of FLEURS `parquet-data/` and VoxPopuli parquet. The audio column must be a struct with `bytes`.
- CV22 folder names (`sv-SE`, `zh-CN`). Missing folders are skipped with a warning.
- soundfile can write Opus (falls back to FLAC otherwise).
- PEFT `modules_to_save` with Parakeet's LSTM decoder. bf16 autocast on MPS.
- llama-server's `input_audio` chat payload for Qwen3-ASR. transcribe-cli `--batch-jsonl` keys `file` and `text`.

## 2026-11-15: CUDA path for a rented GPU
- trainlib.device()/mem_gb(), evaluate.device() now pick cuda → mps → cpu; run.sh uses caffeinate only when present and setup accepts CUDA or Metal.
- Mac still picks mps (checked). CUDA path never run. Originals in runs/pre-cuda-backup/. Rental steps: ~/RENT-GPU.md.
- Last `run.sh all` (2026-11-11) was stopped by hand during baseline; baseline not marked done.
