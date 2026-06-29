"""Backbone encoder/head split and activation extraction.

The predictor g(f(x)) is split into:
  - encoder f: image -> spatial feature map Z (n, p, h, w)
  - head    g: Z -> logits, via global average pooling + the pretrained classifier

Supports the paper's backbones (ResNet34, MobileNetV2) plus ResNet50; all share the
same encoder/head abstraction so the rest of the pipeline is backbone-agnostic.
"""

import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_BACKBONES = {
    "resnet34":     (torchvision.models.resnet34,     torchvision.models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50":     (torchvision.models.resnet50,     torchvision.models.ResNet50_Weights.IMAGENET1K_V2),
    "mobilenet_v2": (torchvision.models.mobilenet_v2, torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2),
}


def load_backbone(name=None):
    """Load a pretrained backbone and its matching preprocessing transform."""
    from config import CONFIG
    name = name or CONFIG["backbone"]
    ctor, weights = _BACKBONES[name]
    model = ctor(weights=weights).to(DEVICE).eval()
    return model, weights.transforms()


def _is_mobilenet(model):
    return isinstance(model, torchvision.models.MobileNetV2)


def encoder(model, x):
    """f: input images -> spatial feature map Z (n, p, h, w)."""
    if _is_mobilenet(model):
        return model.features(x)                         # (n, 1280, 7, 7)
    # ResNet family
    x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
    x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
    return x                                              # (n, 512|2048, 7, 7)


def classify_pooled(model, a):
    """g restricted to its final layer: globally-pooled features a (n, p) -> logits.

    Grad-enabled and architecture-aware (used directly, and by FACE's KL term).
    """
    if _is_mobilenet(model):
        return model.classifier(a)
    return model.fc(a)


def head(model, z):
    """g: spatial feature map -> logits, via GAP + the pretrained classifier."""
    a = torch.flatten(F.adaptive_avg_pool2d(z, 1), 1)    # global average pooling
    return classify_pooled(model, a)


@torch.no_grad()
def extract_activations(model, transform, images, batch_size=16, desc="activations"):
    """Run the encoder over images, returning Z (n, 2048, 7, 7) on CPU."""
    feats = []
    for i in tqdm(range(0, len(images), batch_size), desc=desc):
        batch = torch.stack([transform(im) for im in images[i:i + batch_size]]).to(DEVICE)
        feats.append(encoder(model, batch).cpu())
    return torch.cat(feats, 0)


@torch.no_grad()
def logits_from_Z(model, Z, batch_size=32):
    """Classifier logits from a (possibly reconstructed) feature map Z (n, p, h, w)."""
    out = []
    for i in range(0, len(Z), batch_size):
        out.append(head(model, Z[i:i + batch_size].to(DEVICE)).cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def logits_from_pooled(model, A_pooled):
    """Classifier logits from already globally-pooled features A (n, p)."""
    return classify_pooled(model, A_pooled.to(DEVICE)).cpu()
