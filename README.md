# Concept Discovery via Language-Guided Matrix Decomposition

Re-implementation of *Interpretable Concept Discovery via Language-Guided Matrix
Decomposition* (ECCV 2026, #11398). It does post-hoc concept discovery: instead of NMF's
free coefficient matrix, it uses **CLIP-guided semantic activations**, so each learned
basis vector maps to a **named, human-readable concept** (`Ā ≈ S Wᵀ`, with `S` fixed from
CLIP and `W` learned).

## Differences from the paper

The method is reproduced faithfully, but at a smaller scale. Known differences:

- **One class, not 40.** We run a single ImageNet class (`tabby cat`, n02123045); the
  paper uses 40.
- **One backbone, not two.** We use **ResNet-34**. The paper also reports **MobileNetV2**.
  (Switch via `CONFIG["backbone"]`; MobileNetV2 and ResNet-50 are wired up.)
- **Concepts come from a file, not an LLM.** The paper generates candidate concepts with
  an LLM. We read them from `concept_vocab.json` and then apply the paper's CLIP filter,
  so runs are reproducible and offline.
- **Some hyperparameters are best guesses.** Values the paper gives only in its
  supplementary (red-circle size, PGD iteration counts, CRAFT depth, FACE settings, the
  C-Ins metric) are marked `[suppl]` in `config.py`. Update them there once the
  supplementary is available.
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

## Secrets / local runs

Never commit keys. Locally, export `HF_TOKEN` as an env var, or copy
`secrets.json.example` → `secrets.json` (gitignored) and fill it in. Artifacts then cache
under the repo (`cache/`, `results/`, `viz/`, also gitignored).

## Concept vocabulary

Concepts are read from `concept_vocab.json` — a table keyed by class name, each mapping to
a long list of candidate concepts. At run time `concepts.get_concepts` filters them down to
`r = 25` in two stages:

- **Lexical (suppl. A1.3):** keep 2–3 word concepts; drop generic filler (`object`,
  `scene`, `animal`, …) and concepts that just repeat the class name — unless they carry a
  visual attribute (color/texture/parts/shape/pose/environment/material).
- **CLIP semantic (suppl. A1.4):** rank the rest by CLIP similarity to the class images,
  then greedily keep diverse ones, dropping any too similar (> 0.80) to one already kept.

To add a class, add a key with at least `r` candidates and set `CONFIG["class_name"]`.
Editing the file auto-invalidates the concept cache.

## Notes

- **Baselines compared fairly.** ICE, CRAFT, and FACE differ only in how the basis `W` is
  learned (NMF / recursive NMF / KL-regularized NMF). All else — backbone, preprocessing,
  splits, `r`, and the non-negative inference used for reconstruction — is identical, as the
  paper requires (Sec 4).
- **C-Ins** is the normalized area under the concept-insertion curve (Sec 4.2): how fast the
  correct-class prediction is restored as top concepts are added.
