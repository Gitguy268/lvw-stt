"""Scoring normalisation, Open ASR Leaderboard style: Whisper's English normaliser for en, basic normaliser elsewhere."""
from __future__ import annotations

from functools import lru_cache

import jiwer

from lvw_stt.langs import CER_LANGS


@lru_cache(maxsize=None)
def _norms():
    from whisper_normalizer.basic import BasicTextNormalizer
    from whisper_normalizer.english import EnglishTextNormalizer
    return EnglishTextNormalizer(), BasicTextNormalizer()


def norm(text: str, lang: str) -> str:
    en, basic = _norms()
    s = (en if lang == "en" else basic)(text or "")
    s = " ".join(s.split())
    return s.replace(" ", "") if lang in CER_LANGS else s


def score(refs: list[str], hyps: list[str], lang: str) -> dict:
    """Corpus-level error rate (%) over one language; empty references are dropped."""
    pairs = [(norm(r, lang), norm(h, lang)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return {"err": float("nan"), "n": 0, "metric": "cer" if lang in CER_LANGS else "wer"}
    r, h = zip(*pairs)
    if lang in CER_LANGS:
        o = jiwer.process_characters(list(r), list(h))
        ref_len = sum(len(x) for x in r)
        metric = "cer"
    else:
        o = jiwer.process_words(list(r), list(h))
        ref_len = sum(len(x.split()) for x in r)
        metric = "wer"
    errs = o.substitutions + o.deletions + o.insertions
    return {"err": round(100 * errs / max(ref_len, 1), 3), "n": len(pairs), "metric": metric,
            "sub": o.substitutions, "del": o.deletions, "ins": o.insertions}
