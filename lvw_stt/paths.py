"""Where every model lives on disk.

Two profiles, picked with the LVW_PROFILE environment variable:
  eu     (default) Phonon-2.5 (25 EU languages) + Qwen3-ASR-0.6B -> LvW-STT-EU-0.6B / LvW-STT-Qwen3-0.6B
  en-nl  Phonon-2.5-English + Dutch                              -> LvW-STT-EN-NL-0.6B
Each profile keeps its own training data, runs, results and step markers, so they never overwrite each other.
"""
import os

from lvw_stt.disk import ROOT

PROFILE = os.environ.get("LVW_PROFILE", "eu")
if PROFILE not in ("eu", "en-nl"):
    raise SystemExit(f"LVW_PROFILE must be 'eu' or 'en-nl', not {PROFILE!r}")

RUNS = ROOT / "runs"
RELEASE = ROOT / "release"
TP = ROOT / "third_party"
DATA = ROOT / "data"
EVAL_DIR = DATA / "eval"

SRC_PHONON = RUNS / "src" / "phonon-2.5"
SRC_PHONON_EN = RUNS / "src" / "phonon-2.5-en"
SRC_QWEN = RUNS / "src" / "qwen3-asr-0.6b"
SRC_PARAKEET_CFG = RUNS / "src" / "parakeet-tdt-0.6b-v3"
PHONON_DENSE = RUNS / "phonon25-dense"          # multilingual Phonon-2.5, rebuilt from its container
PHONON_EN_DENSE = RUNS / "phonon25-en-dense"    # Phonon-2.5-English, rebuilt from its container

# Fermion source models. Revisions and checksums are pinned; fetch.py refuses anything else.
PHONON_SOURCES = {
    "multi": {"repo": "FermionResearch/Phonon-2.5", "rev": "2af4bb8ccceb66a8ce06c4081bb968f1802c1e58",
              "src": SRC_PHONON, "dense": PHONON_DENSE, "tar_sha256": None, "container_sha256": None,
              "check_langs": ("en", "nl", "de", "el", "pl"), "check_max_wer": 30.0},
    "en": {"repo": "FermionResearch/Phonon-2.5-English", "rev": "da200358d638886bc96c758c672ad5568828b119",
           "src": SRC_PHONON_EN, "dense": PHONON_EN_DENSE,
           "tar_sha256": "6bd8119ac5a637e5ece0666f5c1e43dbcd99e1c36b3d0ef3ec5f9b58260a9dd4",
           "container_sha256": "c57119c0f04756932acc1d60d90506346e3708d69f00e654084acc7fe507fe27",
           "check_langs": ("en",), "check_max_wer": 15.0},
}

EU_NAME = "LvW-STT-EU-0.6B"
QW_NAME = "LvW-STT-Qwen3-0.6B"
EN_NL_NAME = "LvW-STT-EN-NL-0.6B"
HF_USER = "Lucasgitguy-co"

MODELS = {
    "phonon-base": ("parakeet", PHONON_DENSE),
    "parakeet-ft": ("parakeet", RELEASE / EU_NAME),
    "parakeet-fast": ("parakeet", RELEASE / f"{EU_NAME}-Fast"),
    "qwen-base": ("qwen", SRC_QWEN),
    "qwen-ft": ("qwen", RELEASE / QW_NAME),
    "qwen-fast": ("qwen", RELEASE / f"{QW_NAME}-Fast"),
    # en-nl profile
    "phonon-en-base": ("parakeet", PHONON_EN_DENSE),
    "phonon-multi": ("parakeet", PHONON_DENSE),
    "en-nl-ft": ("parakeet", RELEASE / EN_NL_NAME),
}

if PROFILE == "en-nl":
    RESULTS = ROOT / "results" / "en-nl"
    TRAIN_DIR = DATA / "train-en-nl"
    PROFILE_RUNS = RUNS / "en-nl"
    BASE_DENSE = PHONON_EN_DENSE
    BASE_SOURCE = "en"
    PROFILE_LANGS = ["en", "nl"]
else:
    RESULTS = ROOT / "results"
    TRAIN_DIR = DATA / "train"
    PROFILE_RUNS = RUNS
    BASE_DENSE = PHONON_DENSE
    BASE_SOURCE = "multi"
    PROFILE_LANGS = None          # use langs.PHONON_LANGS / langs.QWEN_LANGS

# en-nl evaluation sets (all stored under data/eval/ by `python -m lvw_stt.data eval-en-nl`)
ESB_SETS = ["esb-ls-clean", "esb-ls-other", "esb-ami", "esb-earnings22", "esb-gigaspeech", "esb-spgispeech",
            "esb-voxpopuli"]
EN_NL_SETS = ESB_SETS + ["fleurs-test-en-nl", "cv22-test-en-nl", "voxpopuli-test-nl"]
EN_NL_DEV = EVAL_DIR / "dev-en-nl.jsonl"

TRANSCRIBE_CLI = TP / "transcribe.cpp" / "build" / "bin" / "transcribe-cli"
TRANSCRIBE_QUANT = TP / "transcribe.cpp" / "build" / "bin" / "transcribe-quantize"
LLAMA_BIN = TP / "llama.cpp" / "build" / "bin"
