import re

def segment_sentences(text: str) -> list[str]:
    """Segment text into sentences using common delimiters.
    
    Delimiters: 。 ! ? ; ! \n
    """
    if not text:
        return []
    
    # Use lookbehind to keep delimiters
    # pattern: matches any of the delimiters, but keep them at the end of the sentence
    sentences = re.split(r'(?<=[。！？；!?\n])', text)
    # filter empty strings and strip whitespace from each sentence
    return [s.strip() for s in sentences if s.strip()]

def format_indexed_text(sentences: list[str]) -> str:
    """Format a list of sentences into a numbered string for LLM prompts."""
    return "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences))

def decode_indices(sentences: list[str], indices: list[int]) -> str:
    """Reconstruct text from a list of sentence indices (1-based)."""
    # 句子之间用换行分隔，避免多句/多说话人粘连（如“…测试。deepseek: …”）
    return "\n".join(sentences[i-1] for i in indices if 0 < i <= len(sentences))


def normalize_indices(indices) -> list[int]:
    """Expand mixed flat/range sentence indices into a flat 1-based int list.

    LLM 对超长输入会用区间压缩索引（如 [[14, 1475]] 表示 14..1475）；
    扁平数字列表 [1, 2, 3] 按单索引处理。本函数统一展开为扁平 int 列表：
        [1, 2, 3], [[5, 8], 10] -> [1, 2, 3, 5, 6, 7, 8, 10]
    """
    out: list[int] = []
    for v in indices or []:
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out.append(int(v))
        elif isinstance(v, (list, tuple)):
            if not v:
                continue
            start = int(v[0])
            end = int(v[1]) if len(v) >= 2 else start
            if start <= end:
                out.extend(range(start, end + 1))
            else:
                out.extend(range(start, end - 1, -1))
    return out


def segment_text_hits(segment_text: str, event_content: str) -> bool:
    """判断一段原文是否与事件内容重叠（整批合并后按内容归属对象/台账）。

    归一化（去空白）后整段包含，或任一断句被事件内容包含，均视为命中。
    """
    norm_event = "".join(event_content.split())
    norm_seg = "".join(segment_text.split())
    if norm_seg and norm_seg in norm_event:
        return True
    return any(
        s and s in norm_event
        for s in ("".join(x.split()) for x in segment_sentences(segment_text))
    )
