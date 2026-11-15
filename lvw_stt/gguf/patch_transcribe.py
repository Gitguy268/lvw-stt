"""Add LvW-STT profiles to transcribe.cpp's convert-parakeet.py (idempotent). Profiles are keyed by the output
folder name, so models/<key>/<key>-F32.gguf selects them. Both are HF-safetensors ParakeetForTDT exports that need
transformers-style length masking (as Fermion's Phonon-2 port does)."""
from __future__ import annotations

import sys
from pathlib import Path

ANCHOR = "VARIANT_PROFILES: dict[str, dict] = {\n"
PROFILE = '''    "{key}": {{
        "variant": "tdt-0.6b-{key}",
        "display_name": "{name}",
        "version": "v1",
        "size_label": "{size}",
        "head_kind": "tdt",
        "expected_vocab_size": 8192,
        "languages": V3_LANGUAGES,
        "lang_detect": True,
        "source_format": "hf_safetensors",
        "transformers_runtime": True,
        "author": "Lucas van de Weerd",
        "organization": "Lucasgitguy-co",
        "license": "cc-by-4.0",
        "license_name": "Creative Commons Attribution 4.0",
        "license_link": "https://creativecommons.org/licenses/by/4.0/",
    }},
'''


def main(path: str) -> None:
    p = Path(path)
    s = p.read_text()
    if '"lvw-stt-eu-0.6b"' in s:
        print("transcribe.cpp already patched"); return
    assert s.count(ANCHOR) == 1, "convert-parakeet.py layout changed; update the patch"
    add = PROFILE.format(key="lvw-stt-eu-0.6b", name="LvW-STT-EU-0.6B", size="0.6B") + \
          PROFILE.format(key="lvw-stt-eu-0.6b-fast", name="LvW-STT-EU-0.6B-Fast", size="0.5B")
    p.write_text(s.replace(ANCHOR, ANCHOR + add))
    print("patched", p)


if __name__ == "__main__":
    main(sys.argv[1])
