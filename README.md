<div align="center">

# LvW-STT

Scripts to post-train open speech-to-text models on one machine (a 16 GB Apple M-series Mac or a rented CUDA GPU), score them, prune them and export GGUF files.

[Profiles](#profiles) · [Quick start](#quick-start) · [Stages](#stages) · [Gates](#gates) · [Status](#status) · [Contributing](#contributing)

![Status](https://img.shields.io/badge/status-not%20trained%20yet-e17055) ![Python](https://img.shields.io/badge/python-3.12-3776ab) ![Devices](https://img.shields.io/badge/devices-Apple%20Metal%20%7C%20CUDA-636e72) ![License](https://img.shields.io/badge/code-MIT-2d3436)

</div>

## Profiles

| Profile | Base model | Target | Planned output |
|---|---|---|---|
| `eu` (default) | [FermionResearch/Phonon-2.5](https://huggingface.co/FermionResearch/Phonon-2.5) (Parakeet TDT, 25 EU languages) and [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) (30 languages) | better accuracy in every supported language | LvW-STT-EU-0.6B, LvW-STT-Qwen3-0.6B, optional `-Fast` and `-GGUF` repos |
| `en-nl` | FermionResearch/Phonon-2.5-English, plus Dutch | add Dutch without losing English | LvW-STT-EN-NL-0.6B and its GGUF repo |

Training data comes from FLEURS, Common Voice 22 and VoxPopuli, streamed and stored as 32 kbps Opus to save disk space. Evaluation audio is FLAC.

## Quick start

You need [uv](https://docs.astral.sh/uv/), CMake, about 20 GB of free disk space, and either Apple Metal or CUDA.

```sh
git clone https://github.com/Gitguy268/lvw-stt.git
cd lvw-stt
bash run.sh setup        # two venvs, pinned clones and builds of transcribe.cpp, llama.cpp, Qwen3-ASR, model downloads
bash run.sh all          # the whole eu pipeline, resumable; ends with PRIVATE Hugging Face repos
bash run.sh en-nl        # or the English + Dutch profile, start to finish
bash run.sh status       # free disk and finished steps
```

Each stage checks free disk space before it starts and stops cleanly when there isn't enough. Nothing is uploaded except by `publish`, which creates private repos. Making them public is a separate command (`bash run.sh publish --public`).

## Stages

| Command | What it does |
|---|---|
| `bash run.sh setup` | venvs, pinned clones and Metal/CUDA builds, model downloads |
| `bash run.sh prepare-eval` | FLEURS test/dev and CV22 test subsets for scoring (FLAC, about 1.5 GB) |
| `bash run.sh expand` | check that the Phonon-2.5 container expands into a working `ParakeetForTDT` |
| `bash run.sh baseline` | score the stock models (`--full` for the full FLEURS test set) |
| `bash run.sh prepare` | training audio weighted by baseline error |
| `bash run.sh train-parakeet --smoke` | short smoke run, then run without `--smoke` (`--resume` to continue) |
| `bash run.sh train-qwen --smoke` | the same for Qwen3-ASR |
| `bash run.sh export` | merge the LoRA into `release/*/` |
| `bash run.sh evaluate` | score the fine-tuned models and apply the gate |
| `bash run.sh prune-parakeet` / `prune-qwen` | optional pruned and healed Fast variants |
| `bash run.sh gguf` | GGUF ladders, scored one file at a time |
| `bash run.sh publish` | private Hugging Face repos with model cards |

## Gates

A model is only published when it passes its gate:

- **Main (eu):** at least 3% better on average, no language more than 5% worse, and Common Voice 22 not worse.
- **en-nl:** Dutch at least 3% better than Phonon-2.5-English and no more than 5% worse than the 25-language Phonon-2.5, the ESB English average not worse, and no English set more than 3% worse.
- **Fast:** within 5% of the main model and at least 1.2× faster.
- **GGUF:** parity of at least 98% (Parakeet F32) or 90% (Qwen BF16) with the PyTorch model.

<details>
<summary>Pinned sources</summary>

| What | Revision |
|---|---|
| FermionResearch/Phonon-2.5 | `2af4bb8ccceb66a8ce06c4081bb968f1802c1e58` |
| FermionResearch/Phonon-2.5-English | `da200358d638886bc96c758c672ad5568828b119` |
| Qwen/Qwen3-ASR-0.6B | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` |
| nvidia/parakeet-tdt-0.6b-v3 (config and tokenizer only) | `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` |
| handy-computer/transcribe.cpp | `5700f127e38dd53c4735ea23ef743b4b538eacd4` |
| ggml-org/llama.cpp | `a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5` |
| QwenLM/Qwen3-ASR | `7c6daf77a2421100f5fb066495372c00129d39ff` |

</details>

<details>
<summary>Measured on an M4 with 16 GB</summary>

- Parakeet training: 1.2 to 2.8 s per step, 11.3 GB peak.
- Qwen training: 2.4 to 4.4 s per step, 9.7 GB peak (40 s of audio per batch, gradient accumulation 16).
- Phonon expansion check: 7.9% WER on English.
- Baseline results so far are in [`results/`](results).

</details>

## Status

The setup, eval preparation, expansion and both smoke runs have run and been checked. Baseline scoring, full training, export, pruning, GGUF and publishing **have not run yet**, so no LvW-STT model exists on Hugging Face. The CUDA code path has not been run either. [`PROGRESS.md`](PROGRESS.md) is the working log.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

The code is under the [MIT License](LICENSE). Models made with it keep their base model's licence: CC-BY-4.0 for Phonon-derived models and Apache-2.0 for Qwen3-ASR-derived models. The `third_party/` clones are not part of this repository. See [NOTICE](NOTICE).
