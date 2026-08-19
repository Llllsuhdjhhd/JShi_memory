"""Chinese text normalization for deterministic substring matching."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _opencc_t2s():
    try:
        from opencc import OpenCC

        return OpenCC("t2s")
    except ImportError:
        return None


# 无 OpenCC 时的最小兜底（红楼梦 ingest 常见繁体；完整转换仍依赖 opencc）。
_FALLBACK_T2S = str.maketrans(
    {
        "寶": "宝",
        "鳳": "凤",
        "鐘": "钟",
        "並": "并",
        "這": "这",
        "個": "个",
        "裡": "里",
        "來": "来",
        "還": "还",
        "說": "说",
        "問": "问",
        "麼": "么",
        "為": "为",
        "與": "与",
        "給": "给",
        "對": "对",
        "將": "将",
        "會": "会",
        "沒": "没",
        "見": "见",
        "聽": "听",
        "讓": "让",
        "從": "从",
        "當": "当",
        "應": "应",
        "該": "该",
        "領": "领",
        "銀": "银",
        "書": "书",
        "閣": "阁",
        "辦": "办",
        "寧": "宁",
        "國": "国",
        "榮": "荣",
        "賈": "贾",
        "蘇": "苏",
        "帶": "带",
        "靈": "灵",
        "爺": "爷",
        "媽": "妈",
        "頭": "头",
        "兩": "两",
        "們": "们",
        "時": "时",
        "後": "后",
        "裏": "里",
        "體": "体",
        "歡": "欢",
        "臉": "脸",
        "聲": "声",
        "歎": "叹",
        "雖": "虽",
        "卻": "却",
        "豈": "岂",
        "誰": "谁",
        "無": "无",
        "緊": "紧",
        "罷": "罢",
    }
)


def to_simplified(text: str) -> str:
    """Normalize *text* to simplified Chinese for name / alias matching."""
    if not text:
        return ""
    converter = _opencc_t2s()
    if converter is not None:
        return converter.convert(text)
    return text.translate(_FALLBACK_T2S)


def normalize_for_substring_match(text: str) -> str:
    """Lowercase + simplified form used as the haystack for role focus matching."""
    return to_simplified(text).lower()
