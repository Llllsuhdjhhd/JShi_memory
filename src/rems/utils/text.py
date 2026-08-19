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
    return "".join(sentences[i-1] for i in indices if 0 < i <= len(sentences))
