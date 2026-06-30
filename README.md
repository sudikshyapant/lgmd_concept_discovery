# Concept Discovery via Language-Guided Matrix Decomposition

Re-implementation of *Interpretable Concept Discovery via Language-Guided Matrix
Decomposition* (ECCV 2026, #11398). It does post-hoc concept discovery: instead of NMF's
free coefficient matrix, it uses **CLIP-guided semantic activations**, so each learned
basis vector maps to a **named, human-readable concept** (`Ā ≈ S Wᵀ`, with `S` fixed from
CLIP and `W` learned).

## Differences from the paper

The method is reproduced faithfully, but at a reduced scope. Known differences:

- **The 40 classes are a reconstruction.** The paper evaluates on 40 ImageNet-1k classes
  but never lists them. We cover all 40 in `concept_vocab.json`, chosen to mirror the
  paper's category structure — 10 semantic categories (birds, insects, mammals, marine,
  reptiles/amphibians, food, indoor objects, nature scenes, infrastructure, vehicles) — and
  pinning the three classes the paper names indirectly (cat, ambulance, acoustic guitar). The
  exact class names, images, and LLM-generated concept vocabularies therefore differ from the
  paper's; figures are faithful analogues, not verbatim reproductions.
- **One backbone.** We use **ResNet-34** only. The paper also reports **MobileNetV2**; we
  don't run it here.
- **Concepts come from a file, not an LLM.** The paper generates candidate concepts with
  an LLM. We read them from `concept_vocab.json` and then apply the paper's CLIP filter,
  so runs are reproducible and offline.
- **No Places365 extension.** The paper adds an 8-class Places365 scene experiment (A3);
  we keep scope to ImageNet only and don't run it.
- **A few numeric hyperparameters are still best guesses.** The supplementary material
  *is* incorporated — its concept-filtering rules (A1.3/A1.4) and CLIP setup with red-circle
  prompting (A1.6) follow it. What remains are a handful of values it describes only qualitatively
  (red-circle radius and stroke width, PGD iteration counts, CRAFT depth, FACE settings,
  the C-Ins metric choice): these stay marked `[suppl]` in `config.py` as informed
  defaults you can tune.
- **Backbone and CLIP share one image crop.** Both read the same 224×224 crop so their
  7×7 grids line up cell-for-cell. This keeps the concept maps spatially accurate.

Everything else — the CLIP similarity maps `S`, the decomposition, `r = 25` concepts, the
ICE/CRAFT/FACE baselines, and the metrics (Acc, C-Ins, MSE) — follows the paper.

## Running on Google Colab

1. Push this repo to GitHub.
2. In a Colab notebook:
   ```python
   !git clone https://github.com/<you>/lgmd.git
   %cd lgmd
   !pip -q install -r requirements.txt
   ```
3. Add `HF_TOKEN` in the Colab **Secrets** panel (key icon) — a HuggingFace token that has
   **accepted the gated ImageNet-1k terms**.
4. Open `lgmd.ipynb` and run top to bottom. Importing `config` mounts Google Drive; heavy
   artifacts (activations, `S`, `W`, concepts, results, viz) are cached under `MyDrive/lgmd/`
   and reused on later runs.

## Running the 40-class benchmark

The notebook's numbered cells walk through one class (the *active* class, default
`tabby cat`). To run all 40, use the driver in `src/runner.py` (last notebook section):

```python
import runner
results, agg = runner.run_all()              # all 40 classes, cached per class
results, agg = runner.run_all(["library"])   # a subset
```

`config.CLASSES` is the 40-class registry (name + WNID + category); `config.select_class(name)`
makes a class active. Label indices are resolved at runtime from the dataset's own label
names — `config.imagenet_class_index(name)` — so there is no hardcoded index table to drift
out of sync. Qualitative concept overlays are saved only for `config.FIGURE_CLASSES` — the
classes the paper shows in its figure rows (cat, bald eagle, library, electric guitar) — so
the visualizations stay comparable to the paper.

## Secrets / local runs

Never commit keys. Locally, export `HF_TOKEN` as an env var, or copy
`secrets.json.example` → `secrets.json` (gitignored) and fill it in. Artifacts then cache
under the repo (`cache/`, `results/`, `viz/`, also gitignored).

## Concept vocabulary

Concepts are read from `concept_vocab.json` — a table keyed by snake_case class identifier
(e.g. `great_white_shark`), each mapping to a long list of candidate concepts. At run time
`concepts.get_concepts` filters them down to `r = 25` in two stages:

- **Lexical (suppl. A1.3):** keep 2–3 word concepts; drop generic filler (`object`,
  `scene`, `animal`, …) and concepts that just repeat the class name — unless they carry a
  visual attribute (color/texture/parts/shape/pose/environment/material).
- **CLIP semantic (suppl. A1.4):** rank the rest by CLIP similarity to the class images,
  then greedily keep diverse ones, dropping any too similar (> 0.80) to one already kept.

To add a class, add a snake_case key with at least `r` candidates and set
`CONFIG["class_name"]` — the human-readable name (e.g. `"great white shark"`) resolves to its
key automatically. Editing the file auto-invalidates the concept cache.

## Notes

- **Baselines compared fairly.** ICE, CRAFT, and FACE differ only in how the basis `W` is
  learned (NMF / recursive NMF / KL-regularized NMF). All else — backbone, preprocessing,
  splits, `r`, and the non-negative inference used for reconstruction — is identical, as the
  paper requires (Sec 4).
- **C-Ins** is the normalized area under the concept-insertion curve (Sec 4.2): how fast the
  correct-class prediction is restored as top concepts are added.
