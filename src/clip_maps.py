"""Localized CLIP similarity maps S via red-circle prompting (paper Sec 3.3).

For each image and each cell of an (h x w) grid aligned to the encoder resolution,
we overlay a small red circle, encode the whole image with CLIP, and take the cosine
similarity to each concept's text embedding. The result is the fixed, language-aligned
coefficient matrix S used in the reconstruction A_bar ~ S W^T.
"""

import torch
import torch.nn.functional as F
from PIL import ImageDraw
from tqdm.auto import tqdm

from config import CONFIG
from data_utils import clip_preprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CLIP:
    """Thin wrapper around a HF CLIP model exposing image/text embedding helpers."""

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        name = CONFIG["clip_model"]
        self.model = CLIPModel.from_pretrained(name).to(DEVICE).eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    @torch.no_grad()
    def embed_text(self, texts):
        inp = self.proc(text=list(texts), return_tensors="pt", padding=True).to(DEVICE)
        e = self.model.get_text_features(**inp)
        return F.normalize(e, dim=-1).cpu()

    @torch.no_grad()
    def embed_images(self, pil_images):
        # Images are already 224x224 (CLIP geometric preprocessing done upstream),
        # so the processor's resize/center-crop are effectively no-ops.
        inp = self.proc(images=list(pil_images), return_tensors="pt").to(DEVICE)
        e = self.model.get_image_features(**inp)
        return F.normalize(e, dim=-1).cpu()


def _grid_variants(img224, grid, radius):
    """Return grid*grid copies of the image, each with a circle marker at one cell center.

    Cells are visited row-major (i over h, j over w) so variant index = i*grid + j,
    matching the row-major spatial unfolding of the encoder feature map.
    """
    cell = img224.size[0] / grid
    width, color = CONFIG["circle_width"], CONFIG["circle_color"]
    variants = []
    for i in range(grid):           # row  (h)
        for j in range(grid):       # col  (w)
            v = img224.copy()
            cx, cy = (j + 0.5) * cell, (i + 0.5) * cell
            ImageDraw.Draw(v).ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                outline=color, width=width,
            )
            variants.append(v)
    return variants


def build_S(images, concepts, clip):
    """Build the semantic activation matrix S of shape (n*h*w, r).

    Each row is a spatial location; each column is a named concept. Values are the
    (non-negative) CLIP image-text cosine similarities under red-circle localization.
    """
    grid, radius = CONFIG["grid"], CONFIG["circle_radius"]
    prompts = [CONFIG["prompt_template"].format(c) for c in concepts]
    text_emb = clip.embed_text(prompts)                 # (r, d)
    rows = []
    for img in tqdm(images, desc="CLIP similarity maps S"):
        variants = _grid_variants(clip_preprocess(img), grid, radius)
        img_emb = clip.embed_images(variants)           # (h*w, d)
        sim = img_emb @ text_emb.T                       # (h*w, r) cosine similarity
        rows.append(sim)
    return torch.cat(rows, 0).clamp(min=0)               # S in R_+^{(nhw) x r}
