# LGMD — Language-Guided Matrix Decomposition

Faithful re-implementation of *Interpretable Concept Discovery via Language-Guided
Matrix Decomposition* (ECCV 2026, #11398). Post-hoc concept discovery that replaces
the latent coefficient matrix of NMF with **CLIP-guided semantic activations**, so each
learned basis vector corresponds to a **named, human-interpretable concept**.

**Scope of this implementation:** a single ImageNet-1k class (`tabby cat`, n02123045)
with a **ResNet-50** backbone — the only intentional deviation from the paper (which uses
40 classes). Methodology, hyperparameters (`r = 25` concepts), baselines, and tests match
the paper.

## Running on Google Colab (mounted via GitHub)

1. Push this repo to GitHub.
2. In a Colab notebook:
   ```python
   !git clone https://github.com/<you>/lgmd.git
   %cd lgmd
   !pip -q install -r requirements.txt
   ```
3. Add secrets in the Colab **Secrets** panel (key icon):
   - `HF_TOKEN` — HuggingFace token that has **accepted the gated ImageNet-1k terms**
   - `OPENAI_API_KEY` — for concept generation
4. Open `lgmd.ipynb` and run top-to-bottom. Importing `config` auto-mounts Google Drive;
   all constant artifacts (activations, `S`, `W`, concepts, results, viz) persist under
   `MyDrive/lgmd/` and are reused on later runs.

## Secrets / local runs

Never commit keys. Locally, either export env vars (`HF_TOKEN`, `OPENAI_API_KEY`) or copy
`secrets.json.example` → `secrets.json` (gitignored) and fill it in. Artifacts then cache
under the repo (`cache/`, `results/`, `viz/`, also gitignored).

## Notes

- ICE/CRAFT/FACE all learn a basis `W` on training activations; validation reconstruction
  reuses LGMD's non-negative inference so methods are compared identically — matching the
  paper's "identical backbones, preprocessing, data splits, and concept counts r" (Sec 4).
  
  The only difference is how `W` is learned (ICE = NMF, CRAFT = recursive NMF, FACE = KL-reg NMF).
  
- **C-Ins** is reported as the normalized area under the concept-insertion curve (Sec 4.2):
  how fast the true-class prediction is restored as top-ranked concepts are added.
  
- Adding the **MobileNet** backbone or **Places365** only requires extending `CONFIG`.
