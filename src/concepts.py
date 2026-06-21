"""Class-specific concept vocabulary generation via OpenAI + CLIP-based filtering.

Following the paper (Sec 3.3): prompt an LLM for visually-grounded concepts, then
filter out (i) generic filler terms, (ii) concepts overlapping the class name, and
(iii) near-duplicates in CLIP text-embedding space. Result: exactly r concepts.
"""

import json
import os
import re

from config import CONFIG, cache_name, get_secret


def _generate_raw(cls, n):
    """Ask the LLM for ~n candidate concepts (over-generate to survive filtering)."""
    from openai import OpenAI
    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
    kwargs = {
        "model": CONFIG["openai_model"],
        "messages": [{"role": "user", "content": CONFIG["concept_prompt"].format(n=n, cls=cls)}],
    }
    if CONFIG["openai_temperature"] is not None:        # some models accept only the default temperature
        kwargs["temperature"] = CONFIG["openai_temperature"]
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content
    text = text[text.find("["): text.rfind("]") + 1]    # isolate the JSON array
    return [c.strip().lower() for c in json.loads(text) if c.strip()]


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
    raw = _generate_raw(cls, int(r * CONFIG["concept_overgen_factor"]))   # over-generate for headroom
    concepts = _filter(raw, cls, clip, CONFIG["dedup_threshold"], r)
    if len(concepts) < r:
        raise RuntimeError(f"Only {len(concepts)}/{r} concepts survived filtering; "
                           f"re-run or raise the over-generation factor.")

    with open(path, "w") as f:
        json.dump(concepts, f, indent=2)
    print(f"[cache] saved concepts.json ({len(concepts)} concepts)")
    return concepts
