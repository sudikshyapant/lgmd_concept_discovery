"""Drive the LGMD pipeline across the 40-class ImageNet benchmark.

The backbone and CLIP are class-independent, so they are loaded once and reused. For
each class we select it (config.select_class sets class_name / synset / index), then run
data -> activations -> concepts -> S -> W -> inference -> metrics + baselines. Every heavy
artifact is cached per class via cache_name() (the "data"/"con" groups key on the active
class). Qualitative overlays are produced only for FIGURE_CLASSES, mirroring the paper's
figures (cat, bald eagle, library, electric guitar).

Usage (from the notebook, after the sys.path setup cell):
    import runner
    results, agg = runner.run_all()                 # all 40 classes
    results, agg = runner.run_all(["tabby cat"])    # a subset
"""

import os

import utils
import data_utils
import model_utils
import concepts as concept_mod
import clip_maps
import lgmd
import baselines
import metrics
import viz
from config import CONFIG, CLASSES, FIGURE_CLASSES, cache_name, select_class

DEVICE = model_utils.DEVICE


def run_class(name, model, transform, clip, make_figures=False):
    """Run the full LGMD pipeline for one class and return its metrics + concepts.

    `model`, `transform`, `clip` are the shared (class-independent) backbone and CLIP.
    Set `make_figures` to also save concept-overlay images for this class.
    """
    select_class(name)                                  # sets class_name / synset / index
    CDIR, RDIR, bb = CONFIG["cache_dir"], CONFIG["results_dir"], CONFIG["backbone"]
    label = CONFIG["class_index"]

    # 1. data — collect this class's images (cached implicitly by the stream stop)
    images = data_utils.load_class_images(CONFIG["n_train"] + CONFIG["n_val"])
    train_imgs, val_imgs = data_utils.make_splits(images, CONFIG["n_train"], CONFIG["n_val"])

    # 2. encoder activations (cache key: data + model)
    Z_train = utils.cached(os.path.join(CDIR, cache_name("Z_train", ".pt", "data", "model")),
        lambda: model_utils.extract_activations(model, transform, train_imgs, desc=f"{name} train act"))
    Z_val = utils.cached(os.path.join(CDIR, cache_name("Z_val", ".pt", "data", "model")),
        lambda: model_utils.extract_activations(model, transform, val_imgs, desc=f"{name} val act"))
    A_train = lgmd.unfold(Z_train)

    # 3. concepts — two-stage filter to exactly r (cache key: con)
    concept_list = concept_mod.get_concepts(clip, images=train_imgs)
    assert len(concept_list) == CONFIG["r"], (name, len(concept_list))

    # 4. localized CLIP similarity maps S (cache key: data + con + clip)
    S_train = utils.cached(os.path.join(CDIR, cache_name("S_train", ".pt", "data", "con", "clip")),
        lambda: clip_maps.build_S(train_imgs, concept_list, clip))

    # 5. fit semantic concept basis W via PGD (cache key: data + model + con + clip + pgd)
    W = utils.cached(os.path.join(CDIR, cache_name("W", ".pt", "data", "model", "con", "clip", "pgd")),
        lambda: lgmd.fit_basis(A_train, S_train))

    # 6. inference on correctly-classified val samples (Sec 4)
    orig_logits_full = model_utils.logits_from_Z(model, Z_val)
    keep = orig_logits_full.argmax(-1) == label
    Z_val = Z_val[keep]
    val_imgs = [im for im, k in zip(val_imgs, keep.tolist()) if k]
    A_val = lgmd.unfold(Z_val)
    orig_logits = orig_logits_full[keep]
    S_hat = lgmd.infer(A_val, W)
    A_hat = lgmd.reconstruct(S_hat, W, Z_val.shape)

    def _lgmd_metrics():
        recon_logits = model_utils.logits_from_Z(model, A_hat)
        return {
            **metrics.predictive_preservation(orig_logits, recon_logits, label),
            "kl": metrics.kl_logits(orig_logits, recon_logits),
            "recon_err": metrics.recon_error(A_val, lgmd.unfold(A_hat)),
        }

    lgmd_metrics = utils.cached_json(
        os.path.join(RDIR, cache_name("lgmd_metrics", ".json",
                                      "data", "model", "con", "clip", "pgd", "infer")),
        _lgmd_metrics)

    # 7. baseline comparison: ICE / CRAFT / FACE vs LGMD (Acc + C-Ins)
    R = CONFIG["r"]
    face_head = lambda a: model_utils.classify_pooled(model, a.to(DEVICE))
    head_fn = lambda Z: model_utils.logits_from_Z(model, Z)

    def _comparison():
        bases = {
            "ICE":   baselines.fit_ice(A_train, R),
            "CRAFT": baselines.fit_craft(A_train, R),
            "FACE":  baselines.fit_face(A_train, R, face_head, Z_train.shape),
            "LGMD":  W,
        }
        out = {}
        for bn, Wb in bases.items():
            Sb = lgmd.infer(A_val, Wb)
            Ab = lgmd.reconstruct(Sb, Wb, Z_val.shape)
            lg = model_utils.logits_from_Z(model, Ab)
            cur = metrics.faithfulness_curves(Sb, Wb, Z_val.shape, head_fn, label)
            pp = metrics.predictive_preservation(orig_logits, lg, label)
            out[bn] = {
                "Acc": pp["recon_acc"],
                "C-Ins": metrics.insertion_auc(cur["insertion"]),
                "agreement": pp["agreement"],
                "kl": metrics.kl_logits(orig_logits, lg),
                "recon_err": metrics.recon_error(A_val, lgmd.unfold(Ab)),
            }
        return out

    comparison = utils.cached_json(
        os.path.join(RDIR, cache_name("comparison", ".json",
                                      "data", "model", "con", "clip", "pgd", "infer", "base", "cins")),
        _comparison)

    # 8. qualitative overlays — only for the paper's figure classes
    if make_figures:
        safe = name.replace(" ", "_")
        viz.save_concept_overlays(val_imgs, S_hat, concept_list,
                                  out_name=f"overlays_{safe}_{bb}.png", max_images=4)

    return {
        "class": name, "index": label, "synset": CONFIG["class_synset"],
        "concepts": concept_list, "lgmd": lgmd_metrics, "comparison": comparison,
    }


def aggregate(results):
    """Mean Acc / C-Ins per method across the run's classes."""
    methods = ["ICE", "CRAFT", "FACE", "LGMD"]
    agg = {m: {} for m in methods}
    for m in methods:
        for k in ("Acc", "C-Ins"):
            vals = [r["comparison"][m][k] for r in results.values() if m in r["comparison"]]
            agg[m][k] = sum(vals) / len(vals) if vals else None
    return agg


def run_all(classes=None, figure_classes=None):
    """Run the pipeline over `classes` (default: all 40). Returns (results, aggregate).

    Loads the backbone + CLIP once and reuses them. Overlays are saved only for classes
    in `figure_classes` (default: FIGURE_CLASSES).
    """
    classes = classes if classes is not None else [c["name"] for c in CLASSES]
    figset = set(figure_classes if figure_classes is not None else FIGURE_CLASSES)

    model, transform = model_utils.load_backbone()
    clip = clip_maps.CLIP()

    results = {}
    for i, name in enumerate(classes, 1):
        res = run_class(name, model, transform, clip, make_figures=name in figset)
        results[name] = res
        print(f"[{i}/{len(classes)}] {name}: LGMD={res['lgmd']}")
    return results, aggregate(results)
