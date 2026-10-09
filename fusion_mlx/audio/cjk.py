# SPDX-License-Identifier: Apache-2.0
"""CJK text helpers for TTS (#1085).

Kokoro is language-blind at the model level — the G2P frontend is chosen by
``lang_code``, and the default English pipeline ("a") phonemizes CJK text
through espeak, producing gibberish. These helpers detect CJK text, normalize
common K-12 math notation before synthesis, and resolve a Mandarin voice when
the caller did not specify one.
"""

import logging
import re

logger = logging.getLogger(__name__)

_KOKORO_MANDARIN_VOICES = (
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunyang",
)

_KOKORO_MANDARIN_DEFAULT = "zf_xiaoxiao"

_DIGIT_TO_CN = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}

_SYMBOL_TO_CN = {
    "π": "圆周率",
    "≈": "约等于",
    "×": "乘以",
    "÷": "除以",
    "°C": "摄氏度",
    "°": "度",
    "+": "加",
    "-": "减",
    "=": "等于",
    "∞": "无穷大",
    "√": "根号",
    "∑": "求和",
    "≠": "不等于",
    "≤": "小于等于",
    "≥": "大于等于",
}


def is_cjk_text(text: str) -> bool:
    if not text:
        return False
    return any("一" <= ch <= "鿿" for ch in text)


def _digits_to_cn(s: str) -> str:
    return "".join(_DIGIT_TO_CN.get(c, c) for c in s)


def _number_to_cn(n: int) -> str:
    # Convert an integer (0-9999) to spoken Mandarin Chinese.
    if n == 0:
        return "零"
    cn = ""
    if n >= 1000:
        cn += _DIGIT_TO_CN[str(n // 1000)] + "千"
        n %= 1000
    if n >= 100:
        cn += _DIGIT_TO_CN[str(n // 100)] + "百"
        n %= 100
    elif cn and n > 0:
        cn += "零"
    if n >= 10:
        tens = n // 10
        if tens != 1 or not cn:
            cn += _DIGIT_TO_CN[str(tens)]
        cn += "十"
        n %= 10
    elif cn and n > 0:
        cn += "零"
    if n > 0:
        cn += _DIGIT_TO_CN[str(n)]
    return cn


def normalize_cjk_text(text: str) -> str:
    if not text:
        return text
    out = text
    # Percent: 56% → 百分之五十六
    out = re.sub(r"(\d+)%", lambda m: "百分之" + _number_to_cn(int(m.group(1))), out)
    # Fraction: 3/4 → 四分之三 (only for small denominators to avoid false positives on dates)
    out = re.sub(
        r"(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)",
        lambda m: _number_to_cn(int(m.group(2)))
        + "分之"
        + _number_to_cn(int(m.group(1))),
        out,
    )
    # Decimal: 3.14 → 三点一四 (integer part via _number_to_cn, fractional part digit-by-digit)
    out = re.sub(
        r"(\d+)\.(\d+)",
        lambda m: _number_to_cn(int(m.group(1))) + "点" + _digits_to_cn(m.group(2)),
        out,
    )
    # Math symbols
    for sym, cn in _SYMBOL_TO_CN.items():
        out = out.replace(sym, cn)
    return out


def is_mandarin_voice(voice: str | None) -> bool:
    if not voice:
        return False
    return voice.startswith("zf_") or voice.startswith("zm_")


def resolve_mandarin_voice(voice: str | None) -> str:
    if is_mandarin_voice(voice):
        return voice  # type: ignore[return-value]
    logger.info(
        "CJK text detected with non-Mandarin voice %r, overriding to %s",
        voice,
        _KOKORO_MANDARIN_DEFAULT,
    )
    return _KOKORO_MANDARIN_DEFAULT


def kokoro_voice_capabilities() -> list[dict]:
    return [
        {
            "voice": v,
            "languages": ["zh"],
            "gender": "female" if v.startswith("zf") else "male",
        }
        for v in _KOKORO_MANDARIN_VOICES
    ]
