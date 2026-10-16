from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path

CATEGORIES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}


def read_dataset(path: Path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def sessions(sample):
    conv = sample["conversation"]
    result = []
    for key, turns in conv.items():
        match = re.fullmatch(r"session_(\d+)", key)
        if match and turns:
            index = int(match[1])
            date = datetime.strptime(conv[f"session_{index}_date_time"], "%I:%M %p on %d %B, %Y")
            result.append((index, date, turns))
    return sorted(result, key=lambda s: (s[1], s[0]))


def questions(sample, count: int, seed: int):
    """Round-robin category stratification, preserving original question indices."""
    rng = random.Random(seed)
    pools = {}
    for index, qa in enumerate(sample["qa"]):
        pools.setdefault(int(qa["category"]), []).append((index, qa))
    for pool in pools.values():
        rng.shuffle(pool)
    selected = []
    while len(selected) < count and any(pools.values()):
        for category in sorted(pools):
            if pools[category] and len(selected) < count:
                selected.append(pools[category].pop())
    return selected


def question_id(sample, index):
    return f"{sample['sample_id']}:q{index}"


def turn_text(turn):
    text = turn["text"]
    if turn.get("blip_caption"):
        text += " [Image caption: " + turn["blip_caption"] + "]"
    return text


def render_question(qa, seed):
    """LoCoMo category 5 is a choice task, not a free-form unanswerable question.

    Some raw questions have true third answers. Official inputs offer the supplied
    adversarial distractor versus 'Not mentioned in the conversation'.
    """
    if int(qa["category"]) != 5:
        return qa["question"], {}
    distractor = qa.get("adversarial_answer") or qa.get("answer")
    if not isinstance(distractor, str) or not distractor.strip():
        raise ValueError("Adversarial choice task has no distractor")
    values = ["Not mentioned in the conversation", distractor]
    random.Random(seed).shuffle(values)
    options = dict(zip(("a", "b"), values))
    return (qa["question"] + f" Select the correct answer: (a) {options['a']} (b) {options['b']}.", options)


def canonical_answer(prediction, options):
    """Decode LoCoMo's single-letter / parenthesized choices before scoring."""
    if not options:
        return prediction
    value = prediction.strip().lower()
    if len(value) == 1:
        return options["a" if "a" in value else "b"]
    if len(value) == 3:
        return options["a" if "(a)" in value else "b"]
    return prediction
