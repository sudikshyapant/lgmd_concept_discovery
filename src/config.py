"""Central configuration for the LGMD project.

A single CONFIG dict holds every hyperparameter. The module auto-detects whether
it runs on Google Colab and, if so, mounts Google Drive so that all heavy /
constant artifacts (activations, CLIP maps, learned basis, results, visualizations)
persist across sessions — while the code itself is re-cloned from GitHub each time.
"""

import hashlib
import json
import os

_THIS = os.path.dirname(os.path.abspath(__file__))   # .../lgmd/src
REPO_ROOT = os.path.dirname(_THIS)                   # .../lgmd


def _in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


IN_COLAB = _in_colab()

# --- Persistent storage root -------------------------------------------------
# On Colab everything persistent lives on Drive; locally it lives in the repo.
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/lgmd"
else:
    BASE_DIR = REPO_ROOT

CACHE_DIR = os.path.join(BASE_DIR, "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VIZ_DIR = os.path.join(BASE_DIR, "viz")
for _d in (CACHE_DIR, RESULTS_DIR, VIZ_DIR):
    os.makedirs(_d, exist_ok=True)


# Stored concept vocabulary: a JSON table keyed by class name (replaces LLM
# generation). Travels with the repo so runs are reproducible and offline. Each
# class maps to candidate concepts (over-provided; filtered down to r in concepts.py).
CONCEPT_VOCAB_PATH = os.path.join(REPO_ROOT, "concept_vocab.json")


def _vocab_hash():
    """Short hash of the vocabulary file so edits to it invalidate concept caches."""
    if not os.path.exists(CONCEPT_VOCAB_PATH):
        return "missing"
    with open(CONCEPT_VOCAB_PATH, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


# Generic filler terms removed during concept filtering (rule i; suppl. A1.3 lists
# "animal", "object", "scene" as examples).
_FILLER_TERMS = ["object", "scene", "thing", "things", "background",
                 "image", "photo", "picture", "stuff", "item", "animal"]

# Visual-attribute lexicon (suppl. A1.3): a concept carrying any of these words is
# preserved even when it partially overlaps the class name. Representative words
# across the categories the supplementary names (color / texture / parts / shape /
# pose-motion / environment / material).
_ATTRIBUTE_TERMS = [
    # color
    "yellow", "black", "white", "brown", "orange", "gray", "grey", "red",
    "blue", "green", "golden", "amber", "pink", "tan",
    # texture
    "furry", "fluffy", "shiny", "glossy", "smooth", "rough", "striped",
    "spotted", "mottled", "speckled",
    # parts
    "tail", "wings", "ears", "eyes", "paws", "legs", "beak", "fur", "whiskers",
    "claws", "nose", "teeth", "pupils", "irises", "markings",
    # shape
    "long", "short", "round", "curved", "pointed", "flat", "thin", "thick",
    # pose / motion
    "running", "flying", "sitting", "standing", "jumping", "perched", "walking",
    "crouching", "stalking",
    # environment
    "forest", "ocean", "grass", "indoor", "outdoor", "water", "sky", "tree", "ground",
    # material
    "metal", "wood", "stone", "glass", "plastic", "leather", "brick",
]


# NOTE: keys tagged [suppl] are NOT specified in the paper body — their exact values
# live in the paper's supplementary material. The values here are informed defaults;
# change them in this one place once the supplementary is available.
CONFIG = {
    # --- data -----------------------------------------------------------------
    # The *active* class. Defaults to tabby cat; call select_class(name) (see CLASSES
    # below) to switch when looping over the 40-class benchmark. class_index is the
    # ImageNet-1k label index, resolved from the dataset's own label names.
    "dataset": "ILSVRC/imagenet-1k",   # https://huggingface.co/datasets/ILSVRC/imagenet-1k (gated)
    "class_synset": "n02123045",   # tabby cat
    "class_index": 281,            # ImageNet-1k label index for "tabby, tabby cat"
    "class_name": "tabby cat",
    "n_train": 100,                # images used to fit the concept basis W
    "n_val": 50,                   # images used for inference + metrics
    "seed": 0,

    # --- backbone (encoder f + classifier head g) -----------------------------
    # Scope: ResNet34 only for now. The paper also reports MobileNetV2; we don't run it.
    "backbone": "resnet34",        # "resnet34"
    "feat_dim": 512,               # p — encoder channels (resnet34 = 512)
    # Backbone preprocessing: "clip_shared_224" shares CLIP's exact 224 crop so encoder
    # feature cells and CLIP red-circle cells cover identical pixels (aligns A_bar <-> S).
    # Part of the activation cache key so changing it recomputes activations.
    "backbone_preprocess": "clip_shared_224",
    "grid": 7,                     # h = w — encoder / CLIP probing grid, 7x7 (suppl. A1.6)

    # --- CLIP (localized similarity maps) -------------------------------------
    "clip_model": "openai/clip-vit-base-patch16",   # CLIP ViT-B/16 (suppl. A1.6); stated open_clip alt: laion2b_s34b_b88k
    "prompt_template": "a photo of {}",              # "a photo of c" text prompt (suppl. A1.4 / A1.6)
    "circle_radius": 16,           # [suppl] red-circle radius (px) — suppl. A1.6 fixes a radius but gives no value
    "circle_width": 3,             # [suppl] red-circle stroke width (px) — "thin" outline, value unspecified
    "circle_color": "red",         # red localization marker (suppl. A1.6)

    # --- concepts -------------------------------------------------------------
    "r": 25,                       # concepts per class (paper fixes r = 25)
    "concept_vocab_path": CONCEPT_VOCAB_PATH,  # stored class -> concepts table (no LLM)
    "concept_vocab_hash": _vocab_hash(),       # content hash for cache invalidation
    "concept_word_min": 2,         # lexical filter: concepts must be 2-3 words (suppl. A1.3)
    "concept_word_max": 3,
    "concept_filler_terms": _FILLER_TERMS,        # generic filler removed, rule i (suppl. A1.3)
    "concept_attribute_terms": _ATTRIBUTE_TERMS,  # attribute words exempt from class-name overlap (suppl. A1.3)
    "concept_proto_images": 100,   # up to N class images for CLIP relevance ranking (suppl. A1.4)
    "dedup_threshold": 0.80,       # CLIP-text cosine sim above which concepts are near-dups (suppl. A1.4)

    # --- optimization ---------------------------------------------------------
    # NOTE: PGD step sizes are NOT free params — the paper fixes them via the spectral
    # norm (Lipschitz constant), so they are computed, not configured here.
    "pgd_iters": 500,              # [suppl] PGD iterations to fit the basis W
    "infer_iters": 50,             # [suppl] PGD refinement iterations at inference

    # --- baselines ------------------------------------------------------------
    "craft_levels": 2,             # [suppl] CRAFT recursive-NMF depth
    "face_lambda": 1.0,            # [suppl] FACE KL-regularization weight
    "face_iters": 300,             # [suppl] FACE optimization iterations
    "face_lr": 1e-2,               # [suppl] FACE Adam learning rate

    # --- metrics --------------------------------------------------------------
    "cins_metric": "prob",         # [suppl] C-Ins 'model performance': "prob" | "accuracy"

    # --- paths ----------------------------------------------------------------
    "cache_dir": CACHE_DIR,
    "results_dir": RESULTS_DIR,
    "viz_dir": VIZ_DIR,
}


# Cache invalidation: each artifact's filename embeds a short hash of the CONFIG
# values it depends on, so changing any of those values rebuilds only the affected
# caches. Call sites name the dependency groups (see below) that apply.
_CACHE_DEPS = {
    "data":  ["class_index", "n_train", "n_val", "seed"],                       # which images
    "model": ["backbone", "backbone_preprocess"],                               # encoder backbone + preprocessing
    "con":   ["concept_vocab_hash", "concept_filler_terms",                     # concept vocab
              "concept_attribute_terms", "concept_word_min", "concept_word_max",
              "concept_proto_images", "dedup_threshold", "r", "class_name", "clip_model"],
    "clip":  ["clip_model", "prompt_template", "circle_radius",                # CLIP maps S
              "circle_width", "circle_color", "grid"],
    "pgd":   ["pgd_iters"],                                                    # basis W fit
    "infer": ["infer_iters"],                                                  # inference
    "base":  ["craft_levels", "face_lambda", "face_iters", "face_lr"],         # baselines
    "cins":  ["cins_metric"],                                                  # C-Ins metric
}


def cache_name(base, ext, *groups):
    """Build a cache filename '<base>_<hash><ext>'.

    The hash covers the CONFIG values in the named dependency groups, so editing any
    of them yields a new filename (and a fresh cache) instead of reusing a stale one.
    """
    keys = sorted({k for g in groups for k in _CACHE_DEPS[g]})
    blob = json.dumps({k: CONFIG[k] for k in keys}, sort_keys=True, default=str)
    digest = hashlib.md5(blob.encode()).hexdigest()[:8]
    return f"{base}_{digest}{ext}"


def get_secret(name):
    """Fetch a secret (API key / token) without ever committing it.

    Lookup order: Colab Secrets -> environment variable -> gitignored secrets.json.
    """
    if IN_COLAB:
        try:
            from google.colab import userdata
            val = userdata.get(name)
            if val:
                return val
        except Exception:
            pass
    if name in os.environ:
        return os.environ[name]
    spath = os.path.join(REPO_ROOT, "secrets.json")
    if os.path.exists(spath):
        with open(spath) as f:
            val = json.load(f).get(name)
            if val:
                return val
    raise KeyError(
        f"Secret '{name}' not found. Set it in Colab Secrets, an env var, "
        f"or a (gitignored) secrets.json at the repo root."
    )


# ---------------------------------------------------------------------------
# 40-class benchmark registry
# ---------------------------------------------------------------------------
# The paper evaluates on 40 ImageNet-1k classes but never lists them; this set is a
# category-stratified reconstruction (10 semantic categories) that pins the classes
# the paper names indirectly (cat, ambulance, electric guitar) and that appear in its
# figures (also bald eagle, library). `name` is the human-readable class (also the
# concept_vocab.json key, snake_cased) and an exact synonym of the ImageNet-1k label,
# so it doubles as the alias used to resolve the label index. `synset` (WNID) is for
# reference/verification.
CLASSES = [
    # birds (5)
    {"name": "goldfinch",          "synset": "n01531178", "category": "birds"},
    {"name": "bald eagle",         "synset": "n01614925", "category": "birds"},
    {"name": "flamingo",           "synset": "n02007558", "category": "birds"},
    {"name": "peacock",            "synset": "n01806143", "category": "birds"},
    {"name": "ostrich",            "synset": "n01518878", "category": "birds"},
    # insects (3)
    {"name": "monarch butterfly",  "synset": "n02279972", "category": "insects"},
    {"name": "ladybug",            "synset": "n02165456", "category": "insects"},
    {"name": "dragonfly",          "synset": "n02268443", "category": "insects"},
    # mammals (8)
    {"name": "tabby cat",          "synset": "n02123045", "category": "mammals"},
    {"name": "golden retriever",   "synset": "n02099601", "category": "mammals"},
    {"name": "African elephant",   "synset": "n02504458", "category": "mammals"},
    {"name": "zebra",              "synset": "n02391049", "category": "mammals"},
    {"name": "red fox",            "synset": "n02119022", "category": "mammals"},
    {"name": "giant panda",        "synset": "n02510455", "category": "mammals"},
    {"name": "lion",               "synset": "n02129165", "category": "mammals"},
    {"name": "Arabian camel",      "synset": "n02437312", "category": "mammals"},
    # marine (2)
    {"name": "goldfish",           "synset": "n01443537", "category": "marine"},
    {"name": "great white shark",  "synset": "n01484850", "category": "marine"},
    # reptiles / amphibians (5)
    {"name": "American alligator", "synset": "n01698640", "category": "reptiles_amphibians"},
    {"name": "green mamba",        "synset": "n01749939", "category": "reptiles_amphibians"},
    {"name": "tree frog",          "synset": "n01644900", "category": "reptiles_amphibians"},
    {"name": "box turtle",         "synset": "n01669191", "category": "reptiles_amphibians"},
    {"name": "common iguana",      "synset": "n01677366", "category": "reptiles_amphibians"},
    # food (4)
    {"name": "pizza",              "synset": "n07873807", "category": "food"},
    {"name": "banana",             "synset": "n07753592", "category": "food"},
    {"name": "cheeseburger",       "synset": "n07697313", "category": "food"},
    {"name": "espresso",           "synset": "n07920052", "category": "food"},
    # indoor objects (2)
    {"name": "electric guitar",    "synset": "n03272010", "category": "indoor_objects"},
    {"name": "analog clock",       "synset": "n02708093", "category": "indoor_objects"},
    # nature scenes (2)
    {"name": "alp",                "synset": "n09193705", "category": "nature_scenes"},
    {"name": "volcano",            "synset": "n09472597", "category": "nature_scenes"},
    # infrastructure (4)
    {"name": "lighthouse",         "synset": "n02814860", "category": "infrastructure"},
    {"name": "castle",             "synset": "n02980441", "category": "infrastructure"},
    {"name": "suspension bridge",  "synset": "n04366367", "category": "infrastructure"},
    {"name": "library",            "synset": "n03661043", "category": "infrastructure"},
    # vehicles (5)
    {"name": "ambulance",          "synset": "n02701002", "category": "vehicles"},
    {"name": "school bus",         "synset": "n04146614", "category": "vehicles"},
    {"name": "sports car",         "synset": "n04285008", "category": "vehicles"},
    {"name": "airliner",           "synset": "n02690373", "category": "vehicles"},
    {"name": "fire engine",        "synset": "n03345487", "category": "vehicles"},
]

_CLASS_BY_NAME = {c["name"]: c for c in CLASSES}

# Qualitative figures use only this small subset — the classes the paper shows in its
# ImageNet figure rows — so our visualizations are directly comparable to the paper.
FIGURE_CLASSES = ["tabby cat", "bald eagle", "library", "electric guitar"]

_LABEL_NAMES = None


def _imagenet_label_names():
    """ImageNet-1k label strings in index order, from the dataset's own metadata.

    Loads only the dataset *info* (no image download). Each label is a comma-separated
    synonym list, e.g. 'tabby, tabby cat'. Cached after first call.
    """
    global _LABEL_NAMES
    if _LABEL_NAMES is None:
        from datasets import load_dataset_builder
        try:
            token = get_secret("HF_TOKEN")
        except KeyError:
            token = None
        builder = load_dataset_builder(CONFIG["dataset"], token=token)
        _LABEL_NAMES = builder.info.features["label"].names
    return _LABEL_NAMES


def imagenet_class_index(name):
    """Resolve the ImageNet-1k label index whose synonym list contains `name` exactly.

    Matches `name` (case-insensitively) against one of the comma-separated synonyms of
    each label, so it is unambiguous and self-validating: raises unless exactly one
    label matches. No hardcoded index table to drift out of sync with the dataset.
    """
    target = name.strip().lower()
    hits = [i for i, label in enumerate(_imagenet_label_names())
            if target in [p.strip().lower() for p in label.split(",")]]
    if len(hits) != 1:
        raise ValueError(
            f"class name {name!r} matched {len(hits)} ImageNet-1k labels "
            f"(need exactly 1); adjust the name in CLASSES to a unique synonym."
        )
    return hits[0]


def select_class(name):
    """Make `name` the active class: set class_name / class_synset / class_index.

    Subsequent cache_name() calls key on these, so each class gets its own caches.
    Returns the resolved class index.
    """
    if name not in _CLASS_BY_NAME:
        raise KeyError(f"{name!r} not in CLASSES. Available: {[c['name'] for c in CLASSES]}")
    entry = _CLASS_BY_NAME[name]
    CONFIG["class_name"] = name
    CONFIG["class_synset"] = entry["synset"]
    CONFIG["class_index"] = imagenet_class_index(name)
    return CONFIG["class_index"]
