"""Language table: code -> FLEURS config, Common Voice folder, VoxPopuli config, Qwen3-ASR name, which model covers it."""
from __future__ import annotations

# code: (fleurs cfg, cv22 folder or None, voxpopuli cfg or None, qwen name or None, in_phonon)
TABLE = {
    "en": ("en_us", "en", "en", "English", True),
    "es": ("es_419", "es", "es", "Spanish", True),
    "fr": ("fr_fr", "fr", "fr", "French", True),
    "de": ("de_de", "de", "de", "German", True),
    "it": ("it_it", "it", "it", "Italian", True),
    "pt": ("pt_br", "pt", None, "Portuguese", True),
    "nl": ("nl_nl", "nl", "nl", "Dutch", True),
    "pl": ("pl_pl", "pl", "pl", "Polish", True),
    "ru": ("ru_ru", "ru", None, "Russian", True),
    "uk": ("uk_ua", "uk", None, None, True),
    "ro": ("ro_ro", "ro", "ro", "Romanian", True),
    "cs": ("cs_cz", "cs", "cs", "Czech", True),
    "sk": ("sk_sk", "sk", "sk", None, True),
    "bg": ("bg_bg", "bg", None, None, True),
    "hr": ("hr_hr", None, "hr", None, True),
    "sv": ("sv_se", "sv-SE", None, "Swedish", True),
    "da": ("da_dk", "da", None, "Danish", True),
    "fi": ("fi_fi", "fi", "fi", "Finnish", True),
    "et": ("et_ee", "et", "et", None, True),
    "lv": ("lv_lv", "lv", None, None, True),
    "lt": ("lt_lt", "lt", "lt", None, True),
    "hu": ("hu_hu", "hu", "hu", "Hungarian", True),
    "sl": ("sl_si", "sl", "sl", None, True),
    "mt": ("mt_mt", "mt", None, None, True),
    "el": ("el_gr", "el", None, "Greek", True),
    # Qwen-only
    "zh": ("cmn_hans_cn", "zh-CN", None, "Chinese", False),
    "yue": ("yue_hant_hk", "yue", None, "Cantonese", False),
    "ar": ("ar_eg", "ar", None, "Arabic", False),
    "id": ("id_id", "id", None, "Indonesian", False),
    "ko": ("ko_kr", "ko", None, "Korean", False),
    "th": ("th_th", "th", None, "Thai", False),
    "vi": ("vi_vn", "vi", None, "Vietnamese", False),
    "ja": ("ja_jp", "ja", None, "Japanese", False),
    "tr": ("tr_tr", "tr", None, "Turkish", False),
    "hi": ("hi_in", "hi", None, "Hindi", False),
    "ms": ("ms_my", None, None, "Malay", False),
    "fil": ("fil_ph", None, None, "Filipino", False),
    "fa": ("fa_ir", "fa", None, "Persian", False),
    "mk": ("mk_mk", "mk", None, "Macedonian", False),
}

# Scored by character error rate (no word boundaries / Qwen's own convention).
CER_LANGS = {"zh", "yue", "ja", "th", "ko"}


def fleurs(code: str) -> str:
    return TABLE[code][0]


def cv(code: str) -> str | None:
    return TABLE[code][1]


def voxpopuli(code: str) -> str | None:
    return TABLE[code][2]


def qwen_name(code: str) -> str | None:
    return TABLE[code][3]


PHONON_LANGS = [c for c, v in TABLE.items() if v[4]]
QWEN_LANGS = [c for c, v in TABLE.items() if v[3]]
ALL_LANGS = list(TABLE)


def for_model(model: str) -> list[str]:
    return PHONON_LANGS if model.startswith(("phonon", "parakeet")) else QWEN_LANGS
