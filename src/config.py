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


# Generic filler terms removed during concept filtering (paper rule i).
_FILLER_TERMS = ["object", "scene", "thing", "things", "background",
                 "image", "photo", "picture", "stuff", "item"]


# NOTE: keys tagged [suppl] are NOT specified in the paper body — their exact values
# live in the paper's supplementary material. The values here are informed defaults;
# change them in this one place once the supplementary is available.
CONFIG = {
    # --- data -----------------------------------------------------------------
    "dataset": "imagenet-1k",
    "class_synset": "n02123045",   # tabby cat
    "class_index": 281,            # ImageNet-1k label index for "tabby, tabby cat"
    "class_name": "tabby cat",
    "n_train": 100,                # images used to fit the concept basis W
    "n_val": 50,                   # images used for inference + metrics
    "seed": 0,

    # --- backbone (encoder f + classifier head g) -----------------------------
    # Paper tables use ResNet34 + MobileNetV2. Switch via this key; feat_dim follows.
    "backbone": "resnet34",        # "resnet34" | "mobilenet_v2" | "resnet50"
    "feat_dim": 512,               # p — encoder channels (resnet34=512, mobilenet_v2=1280, resnet50=2048)
    "grid": 7,                     # h = w — encoder spatial resolution at 224px input

    # --- CLIP (localized similarity maps) -------------------------------------
    "clip_model": "openai/clip-vit-base-patch16",   # [suppl] which CLIP variant
    "prompt_template": "a photo of {}",              # paper's text prompt (Sec 3.3)
    "circle_radius": 16,           # [suppl] red-circle radius (px) for localized prompting
    "circle_width": 3,             # [suppl] red-circle stroke width (px)
    "circle_color": "red",         # [suppl] localization marker color

    # --- concepts -------------------------------------------------------------
    "r": 25,                       # concepts per class (paper fixes r = 25)
    "concept_vocab_path": CONCEPT_VOCAB_PATH,  # stored class -> concepts table (no LLM)
    "concept_vocab_hash": _vocab_hash(),       # content hash for cache invalidation
    "concept_filler_terms": _FILLER_TERMS,   # [suppl] filler words removed (rule i)
    "dedup_threshold": 0.9,        # [suppl] CLIP-text cosine sim above which concepts are near-dups

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
    "act":   ["backbone", "class_index", "n_train", "n_val", "seed"],          # activations
    "con":   ["concept_vocab_hash", "concept_filler_terms",                     # concept vocab
              "dedup_threshold", "r", "class_name", "clip_model"],
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
