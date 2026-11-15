"""Where every model lives on disk."""
from lvw_stt.disk import ROOT

RUNS = ROOT / "runs"
RELEASE = ROOT / "release"
RESULTS = ROOT / "results"
TP = ROOT / "third_party"

SRC_PHONON = RUNS / "src" / "phonon-2.5"
SRC_QWEN = RUNS / "src" / "qwen3-asr-0.6b"
SRC_PARAKEET_CFG = RUNS / "src" / "parakeet-tdt-0.6b-v3"
PHONON_DENSE = RUNS / "phonon25-dense"

EU_NAME = "LvW-STT-EU-0.6B"
QW_NAME = "LvW-STT-Qwen3-0.6B"
HF_USER = "Lucasgitguy-co"

MODELS = {
    "phonon-base": ("parakeet", PHONON_DENSE),
    "parakeet-ft": ("parakeet", RELEASE / EU_NAME),
    "parakeet-fast": ("parakeet", RELEASE / f"{EU_NAME}-Fast"),
    "qwen-base": ("qwen", SRC_QWEN),
    "qwen-ft": ("qwen", RELEASE / QW_NAME),
    "qwen-fast": ("qwen", RELEASE / f"{QW_NAME}-Fast"),
}

TRANSCRIBE_CLI = TP / "transcribe.cpp" / "build" / "bin" / "transcribe-cli"
TRANSCRIBE_QUANT = TP / "transcribe.cpp" / "build" / "bin" / "transcribe-quantize"
LLAMA_BIN = TP / "llama.cpp" / "build" / "bin"
