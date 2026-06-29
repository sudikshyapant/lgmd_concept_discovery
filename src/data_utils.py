"""ImageNet-1k single-class loading and CLIP geometric preprocessing."""

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from config import CONFIG, get_secret


def load_class_images(n_total):
    """Stream ImageNet-1k and collect `n_total` images of the target class.

    Streaming avoids downloading the full dataset: we stop as soon as we have
    enough images whose label equals CONFIG['class_index'].
    """
    token = get_secret("HF_TOKEN")
    ds = load_dataset(CONFIG["dataset"], split="train", streaming=True, token=token)
    target = CONFIG["class_index"]
    images = []
    pbar = tqdm(total=n_total, desc="collecting class images")
    for ex in ds:
        if ex["label"] == target:
            images.append(ex["image"].convert("RGB"))
            pbar.update(1)
            if len(images) >= n_total:
                break
    pbar.close()
    return images


def make_splits(images, n_train, n_val):
    """Train/val split. Images are already class-homogeneous, so a slice suffices."""
    return images[:n_train], images[n_train:n_train + n_val]


def clip_preprocess(img, size=224):
    """CLIP's deterministic geometric preprocessing: resize shortest side + center crop.

    Returns a `size`x`size` RGB PIL image, so we can draw red circles on it before
    handing it to the CLIP normalizer (resize/crop then become no-ops).
    """
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))
