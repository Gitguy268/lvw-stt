#!/bin/bash
# LvW-STT: post-train Phonon-2.5 (Parakeet TDT) and Qwen3-ASR-0.6B on the local Mac, then GGUF + publish.
# Stages run in order; each checks free disk first. Nothing is uploaded except by `publish`.
# Second profile: `bash run.sh en-nl` = Phonon-2.5-English + Dutch -> LvW-STT-EN-NL-0.6B (stages run with LVW_PROFILE=en-nl).
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
STAGE="${1:-help}"; shift || true

PY_PK="$ROOT/envs/parakeet/bin/python"   # transformers 5.22 (ParakeetForTDT + TDT loss)
PY_QW="$ROOT/envs/qwen/bin/python"       # qwen-asr (pins transformers 4.57.6)
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Pinned sources (2026-11-09)
PHONON_REV=2af4bb8ccceb66a8ce06c4081bb968f1802c1e58
PHONON_EN_REV=da200358d638886bc96c758c672ad5568828b119   # FermionResearch/Phonon-2.5-English (pinned 2026-11-15)
QWEN_REV=5eb144179a02acc5e5ba31e748d22b0cf3e303b0
PARAKEET_V3_REV=541d1f99c6b0c3cd0b11a95167540bb8edefd82b
TRANSCRIBE_CPP_SHA=5700f127e38dd53c4735ea23ef743b4b538eacd4
LLAMA_CPP_SHA=a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5
QWEN_ASR_GH_SHA=7c6daf77a2421100f5fb066495372c00129d39ff

guard() { python3 -I "$ROOT/lvw_stt/disk.py" "${1:-0}" "${2:-$STAGE}"; }
# caffeinate keeps a Mac awake; it does not exist on Linux (rented GPU box)
awake() { if command -v caffeinate >/dev/null; then caffeinate -i "$@"; else "$@"; fi; }
need() { [ -x "$1" ] || { echo "Run: bash run.sh setup"; exit 1; }; }

# Profile: eu (default) or en-nl. The en-nl-only stages set it themselves; shared stages read it from the environment.
case "$STAGE" in
  en-nl|fetch-en|expand-en|eval-sets|calibrate|dryrun) export LVW_PROFILE=en-nl ;;
esac
PROFILE="${LVW_PROFILE:-eu}"
case "$PROFILE" in eu|en-nl) ;; *) echo "LVW_PROFILE must be eu or en-nl"; exit 1 ;; esac
eu_only() { [ "$PROFILE" = eu ] || { echo "'$STAGE' belongs to the eu profile (run it without LVW_PROFILE=en-nl)"; exit 1; }; }

clone_at() {  # clone_at <url> <dir> <sha>
  local url="$1" dir="$2" sha="$3"
  if [ ! -d "$dir/.git" ]; then
    git init -q "$dir"; git -C "$dir" remote add origin "$url"
  fi
  git -C "$dir" fetch -q --depth 1 origin "$sha"
  git -C "$dir" checkout -q --force FETCH_HEAD
}

case "$STAGE" in
  en-nl)
    # Phonon-2.5-English + Dutch, start to finish. Resumable: finished steps are marked in runs/en-nl/.done/.
    # Stops at any failed check (calibration, dry run, smoke test, gate). Ends with PRIVATE repos;
    # making them public stays a separate command: LVW_PROFILE=en-nl bash run.sh publish --public
    need "$PY_PK"
    [ -x third_party/transcribe.cpp/build/bin/transcribe-cli ] || { echo "Run: bash run.sh setup"; exit 1; }
    DONE="runs/en-nl/.done"; mkdir -p "$DONE"
    LOG="runs/en-nl/run-$(date +%Y%m%d-%H%M%S).log"
    exec > >(tee -a "$LOG") 2>&1
    step() {  # step <name> <command...>
      local name="$1"; shift
      if [ -f "$DONE/$name" ]; then echo "== $name: done earlier, skipped"; return 0; fi
      echo "== $name: $(date '+%H:%M:%S')"
      if "$@"; then touch "$DONE/$name"
      else echo "== $name FAILED (log: $LOG). Fix the cause, then re-run: bash run.sh en-nl"; exit 1; fi
    }
    resume_flag() { ls -d runs/en-nl/parakeet-ft/ckpt-* >/dev/null 2>&1 && echo --resume || true; }
    me="$ROOT/run.sh"
    step fetch-en         bash "$me" fetch-en
    step patch-transcribe python3 -I lvw_stt/gguf/patch_transcribe.py third_party/transcribe.cpp/scripts/convert-parakeet.py
    step expand-en        bash "$me" expand-en
    step eval-sets        bash "$me" eval-sets
    step calibrate        bash "$me" calibrate
    step dryrun           bash "$me" dryrun
    step prepare          bash "$me" prepare
    step smoke-parakeet   bash "$me" train-parakeet --smoke
    step train-parakeet   bash "$me" train-parakeet $(resume_flag)
    step export           bash "$me" export
    step clean-audio      bash "$me" clean-audio
    step evaluate         bash "$me" evaluate
    step publish          bash "$me" publish          # private repos; the GGUF ladder is built and uploaded here
    echo "== all done. Results: results/en-nl/  Gate: results/en-nl/gate.json  Log: $LOG"
    echo "   Check https://huggingface.co/Lucasgitguy-co/LvW-STT-EN-NL-0.6B (private), then:"
    echo "   LVW_PROFILE=en-nl bash run.sh publish --public"
    ;;
  fetch-en)
    need "$PY_PK"; guard 1
    "$PY_PK" -m lvw_stt.fetch --phonon-en-rev "$PHONON_EN_REV"
    ;;
  expand-en)
    need "$PY_PK"; guard 0.1
    "$PY_PK" -m lvw_stt.parakeet.expand_phonon --source en
    ;;
  eval-sets)
    need "$PY_PK"; guard 1
    "$PY_PK" -m lvw_stt.data eval-en-nl "$@"
    ;;
  calibrate)
    # Stock Phonon-2.5-English must score close to its own model card on our subsets; also makes every baseline.
    need "$PY_PK"; guard 0.3
    "$PY_PK" -m lvw_stt.evaluate --model phonon-en-base --tag baseline "$@"
    "$PY_PK" -m lvw_stt.evaluate --model phonon-multi --tag baseline "$@"
    "$PY_PK" -m lvw_stt.gate calibrate
    ;;
  dryrun)
    need "$PY_PK"
    python3 -I lvw_stt/gguf/patch_transcribe.py third_party/transcribe.cpp/scripts/convert-parakeet.py
    "$PY_PK" -m lvw_stt.dryrun
    ;;
  all)
    eu_only
    # Everything, start to finish. Finished steps are marked in runs/.done/, so re-running resumes.
    # Stops on a failed smoke test; models that fail their gate are simply not uploaded.
    # Ends with PRIVATE repos; making them public stays a separate command: bash run.sh publish --public
    mkdir -p runs/.done
    LOG="runs/all-$(date +%Y%m%d-%H%M%S).log"
    exec > >(tee -a "$LOG") 2>&1
    step() {  # step <name> <command...>
      local name="$1"; shift
      if [ -f "runs/.done/$name" ]; then echo "== $name: done earlier, skipped"; return 0; fi
      echo "== $name: $(date '+%H:%M:%S')"
      "$@" || return 1
      touch "runs/.done/$name"
    }
    resume_flag() { ls -d "runs/$1"/ckpt-* >/dev/null 2>&1 && echo --resume || true; }
    me="$ROOT/run.sh"
    step setup          bash "$me" setup
    step prepare-eval   bash "$me" prepare-eval
    step expand         bash "$me" expand
    step baseline       bash "$me" baseline
    step prepare        bash "$me" prepare
    step smoke-parakeet bash "$me" train-parakeet --smoke
    step smoke-qwen     bash "$me" train-qwen --smoke
    step train-parakeet bash "$me" train-parakeet $(resume_flag parakeet-ft)
    step train-qwen     bash "$me" train-qwen $(resume_flag qwen-ft)
    step export         bash "$me" export
    step evaluate       bash -c "bash '$me' evaluate || echo 'gate: at least one model failed; it will not be uploaded'"
    if [ -d release/LvW-STT-EU-0.6B ]; then
      step prune-parakeet bash "$me" prune-parakeet $(resume_flag parakeet-fast) \
        || echo "prune-parakeet failed: no Fast EU variant this run (re-run to retry)"
    fi
    if [ -d release/LvW-STT-Qwen3-0.6B ]; then
      step prune-qwen     bash "$me" prune-qwen $(resume_flag qwen-fast) \
        || echo "prune-qwen failed: no Fast Qwen variant this run (re-run to retry)"
    fi
    step evaluate-fast  bash -c "bash '$me' evaluate-fast || echo 'fast gate: a Fast variant failed; it will not be uploaded'"
    step clean-audio    bash "$me" clean-audio
    step publish        bash "$me" publish        # GGUF ladders are built, scored and uploaded here, one file at a time
    echo "== all done. Results: results/  Gate: results/gate.json  Log: $LOG"
    echo "   Check the private repos on huggingface.co/Lucasgitguy-co, then: bash run.sh publish --public"
    ;;
  clean-audio)
    # Training audio is no longer needed after training/pruning (the manifest stays for cards and the imatrix).
    if [ "$PROFILE" = en-nl ]; then
      if [ -d data/train-en-nl ]; then find data/train-en-nl -type f \( -name '*.ogg' -o -name '*.flac' \) -delete; fi
      rm -rf runs/en-nl/parakeet-ft/ckpt-*        # resume checkpoints; the chosen adapter stays in best/
    else
      find data/train -type f \( -name '*.ogg' -o -name '*.flac' \) -delete
    fi
    python3 -I lvw_stt/disk.py
    ;;
  setup)
    eu_only
    guard 6 "venvs + builds"
    command -v uv >/dev/null || { echo 'brew install uv'; exit 1; }
    command -v cmake >/dev/null || { echo 'brew install cmake'; exit 1; }
    [ -x "$PY_PK" ] || uv venv -q --python 3.12 envs/parakeet
    [ -x "$PY_QW" ] || uv venv -q --python 3.12 envs/qwen
    uv pip install -q --python "$PY_PK" -r requirements-parakeet.txt
    uv pip install -q --python "$PY_QW" -r requirements-qwen.txt
    mkdir -p third_party runs
    clone_at https://github.com/handy-computer/transcribe.cpp third_party/transcribe.cpp "$TRANSCRIBE_CPP_SHA"
    clone_at https://github.com/ggml-org/llama.cpp third_party/llama.cpp "$LLAMA_CPP_SHA"
    clone_at https://github.com/QwenLM/Qwen3-ASR third_party/Qwen3-ASR "$QWEN_ASR_GH_SHA"
    python3 -I lvw_stt/gguf/patch_transcribe.py third_party/transcribe.cpp/scripts/convert-parakeet.py
    cmake -S third_party/transcribe.cpp -B third_party/transcribe.cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DTRANSCRIBE_BUILD_TOOLS=ON -DTRANSCRIBE_BUILD_TESTS=OFF >/dev/null
    cmake --build third_party/transcribe.cpp/build -j 8 --target transcribe-cli transcribe-quantize >/dev/null
    cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF >/dev/null
    cmake --build third_party/llama.cpp/build -j 8 \
      --target llama-quantize llama-imatrix llama-mtmd-cli llama-server >/dev/null
    uv pip install -q --python "$PY_QW" -e third_party/llama.cpp/gguf-py
    "$PY_PK" -m lvw_stt.fetch --phonon-rev "$PHONON_REV" --qwen-rev "$QWEN_REV" --parakeet-rev "$PARAKEET_V3_REV"
    "$PY_PK" -c 'import torch; ok = torch.cuda.is_available() or torch.backends.mps.is_available(); assert ok, "no GPU (CUDA or Metal) found"; print("cuda" if torch.cuda.is_available() else "mps", "ok", torch.__version__)'
    "$PY_QW" -c 'import qwen_asr, transformers; print("qwen-asr ok, transformers", transformers.__version__)'
    uv pip freeze --python "$PY_PK" > runs/environment-parakeet.txt
    uv pip freeze --python "$PY_QW" > runs/environment-qwen.txt
    echo "setup done"
    ;;
  expand)
    eu_only
    need "$PY_PK"; guard 3
    "$PY_PK" -m lvw_stt.parakeet.expand_phonon "$@"
    ;;
  prepare)
    need "$PY_PK"
    if [ "$PROFILE" = en-nl ]; then
      # 3 GB of Opus (~200 h): 60% Dutch, 40% English replay so English isn't forgotten. Extra options override.
      guard 3.2
      "$PY_PK" -m lvw_stt.data prepare --gb 3.0 --mix en=0.4,nl=0.6 --shares fleurs=0.15,cv22=0.45,voxpopuli=0.40 "$@"
    else
      guard 2
      "$PY_PK" -m lvw_stt.data prepare "$@"
    fi
    ;;
  prepare-eval)
    eu_only
    need "$PY_PK"; guard 2
    "$PY_PK" -m lvw_stt.data eval "$@"
    ;;
  baseline)
    eu_only
    need "$PY_PK"; guard 1
    "$PY_PK" -m lvw_stt.evaluate --model phonon-base --tag baseline "$@"
    "$PY_QW" -m lvw_stt.evaluate --model qwen-base --tag baseline "$@"
    ;;
  train-parakeet)
    need "$PY_PK"
    if [ "$PROFILE" = en-nl ]; then guard 2; else guard 6; fi    # en-nl: audio is already on disk; checkpoints ~1 GB
    awake "$PY_PK" -m lvw_stt.parakeet.train "$@"
    ;;
  train-qwen)
    eu_only
    need "$PY_QW"; guard 4
    awake "$PY_QW" -m lvw_stt.qwen.train "$@"
    ;;
  export)
    "$PY_PK" -m lvw_stt.parakeet.export "$@"
    if [ "$PROFILE" = eu ]; then "$PY_QW" -m lvw_stt.qwen.export "$@"; fi
    ;;
  evaluate)
    if [ "$PROFILE" = en-nl ]; then
      guard 0.3
      "$PY_PK" -m lvw_stt.evaluate --model en-nl-ft --tag ft "$@"
      "$PY_PK" -m lvw_stt.gate en-nl
      exit 0
    fi
    guard 1
    "$PY_PK" -m lvw_stt.evaluate --model parakeet-ft --tag ft "$@"
    "$PY_QW" -m lvw_stt.evaluate --model qwen-ft --tag ft "$@"
    "$PY_PK" -m lvw_stt.gate main
    ;;
  evaluate-fast)
    eu_only
    guard 1
    [ -d release/LvW-STT-EU-0.6B-Fast ] && "$PY_PK" -m lvw_stt.evaluate --model parakeet-fast --tag fast "$@"
    [ -d release/LvW-STT-Qwen3-0.6B-Fast ] && "$PY_QW" -m lvw_stt.evaluate --model qwen-fast --tag fast "$@"
    "$PY_PK" -m lvw_stt.gate fast
    ;;
  prune-parakeet)
    eu_only
    need "$PY_PK"; guard 6
    awake "$PY_PK" -m lvw_stt.parakeet.prune "$@"
    ;;
  prune-qwen)
    eu_only
    need "$PY_QW"; guard 6
    awake "$PY_QW" -m lvw_stt.qwen.prune "$@"
    ;;
  gguf)
    if [ "$PROFILE" = en-nl ]; then guard 2.6; "$PY_PK" -m lvw_stt.gguf.ladder --family en-nl "$@"
    else guard 4; "$PY_QW" -m lvw_stt.gguf.ladder "$@"; fi
    ;;
  publish)
    # Explicit only. Creates private repos and uploads; --public is a separate, later command.
    if [ "$PROFILE" = en-nl ]; then "$PY_PK" -m lvw_stt.publish "$@"; else "$PY_QW" -m lvw_stt.publish "$@"; fi
    ;;
  status)
    echo "profile: $PROFILE"; python3 -I lvw_stt/disk.py
    if [ "$PROFILE" = en-nl ]; then echo "done: $(ls runs/en-nl/.done 2>/dev/null | tr '\n' ' ')"; fi
    sed -n '1,60p' PROGRESS.md
    ;;
  *)
    cat <<'EOF'
One command for everything (resumable; ends with private HF repos):
  bash run.sh all

Stages, in order:
  bash run.sh setup            venvs, pinned clones + Metal builds, model downloads
  bash run.sh prepare-eval     FLEURS test/dev + CV22 test subsets for scoring (FLAC, ~1.5 GB)
  bash run.sh expand           check the Phonon-2.5 container expands to a working ParakeetForTDT
  bash run.sh baseline         score the stock models (add --full for full FLEURS test)
  bash run.sh prepare          training audio weighted by baseline error (FLEURS/CV22/VoxPopuli train, 5 GB)
  bash run.sh train-parakeet --smoke   then without --smoke (resume: --resume)
  bash run.sh train-qwen --smoke       then without --smoke (resume: --resume)
  bash run.sh export           merge LoRA -> release/*/
  bash run.sh evaluate         score fine-tuned models, apply the accuracy gate
  bash run.sh prune-parakeet   optional Fast variant (then: evaluate with --model parakeet-fast --tag fast)
  bash run.sh prune-qwen       optional Fast variant
  bash run.sh gguf             GGUF ladders, scored one file at a time (nothing uploaded)
  bash run.sh publish          explicit: gated models -> private HF repos; --public afterwards
  bash run.sh status           disk + progress

Phonon-2.5-English + Dutch -> LvW-STT-EN-NL-0.6B (one command, resumable; ends with private HF repos):
  bash run.sh en-nl
  Steps: fetch-en, expand-en, eval-sets, calibrate, dryrun (these set the profile themselves), then
  prepare, train-parakeet, export, clean-audio, evaluate, gguf, publish, status: put LVW_PROFILE=en-nl in front.
EOF
    ;;
esac
