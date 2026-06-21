"""Class-specific concept vocabulary from a stored table + CLIP-based filtering.

Following the paper (Sec 3.3): start from a visually-grounded concept vocabulary,
then filter out (i) generic filler terms, (ii) concepts overlapping the class name,
and (iii) near-duplicates in CLIP text-embedding space. Result: exactly r concepts.

The vocabulary is read from a stored JSON table (CONFIG["concept_vocab_path"]),
keyed by class name — no external LLM API is used. Each class maps to a list of
candidate concepts, given either as plain strings or as {"label": ...} objects
(extra fields such as id / category / description are kept in the table but ignored
here). The table is over-provided on purpose so the three filtering rules have
headroom to land on exactly r concepts.
"""

import json
import os
import re

from config import CONFIG, cache_name


def _load_vocab(cls):
    """Load the stored candidate concepts for `cls` (lowercased strings)."""
    path = CONFIG["concept_vocab_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Concept vocabulary table not found at {path}. "
            f"Create it (see concept_vocab.json) keyed by class name."
        )
    with open(path) as f:
        table = json.load(f)
    if cls not in table:
        raise KeyError(
            f"No concept vocabulary for class '{cls}' in {path}. "
            f"Available classes: {sorted(table)}"
        )
    raw = []
    for c in table[cls]:
        label = (c["label"] if isinstance(c, dict) else c).strip().lower()
        if label:
            raw.append(label)
    return raw


def _filter(concepts, cls, clip, threshold, r):
    """Apply the paper's three filtering rules and keep at most r concepts."""
    cls_words = set(re.findall(r"\w+", cls.lower()))
    filler = set(CONFIG["concept_filler_terms"])
    kept, kept_emb = [], []
    for c in concepts:
        words = set(re.findall(r"\w+", c))
        if words & cls_words:               # (ii) class-name overlap / trivial leakage
            continue
        if words & filler:                  # (i) generic filler
            continue
        emb = clip.embed_text([c])[0]       # (iii) CLIP-text near-duplicate
        if any(float(emb @ e) > threshold for e in kept_emb):
            continue
        kept.append(c)
        kept_emb.append(emb)
        if len(kept) == r:
            break
    return kept


def get_concepts(clip):
    """Return exactly r concepts for the target class, cached to JSON on Drive."""
    path = os.path.join(CONFIG["cache_dir"], cache_name("concepts", ".json", "con"))
    if os.path.exists(path):
        with open(path) as f:
            print("[cache] loaded concepts.json")
            return json.load(f)

    cls, r = CONFIG["class_name"], CONFIG["r"]
    raw = _load_vocab(cls)                                  # stored vocabulary (over-provided)
    concepts = _filter(raw, cls, clip, CONFIG["dedup_threshold"], r)
    if len(concepts) < r:
        raise RuntimeError(f"Only {len(concepts)}/{r} concepts survived filtering; "
                           f"add more candidates for '{cls}' in {CONFIG['concept_vocab_path']}.")

    with open(path, "w") as f:
        json.dump(concepts, f, indent=2)
    print(f"[cache] saved concepts.json ({len(concepts)} concepts)")
    return concepts
