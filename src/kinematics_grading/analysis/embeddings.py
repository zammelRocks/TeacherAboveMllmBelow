"""Single, shared ResNet-50 embedding cache.

The original codebase re-implemented this same frozen-ResNet-50 + npz-cache
pattern independently in reverse_corpus_analysis_notebook.ipynb,
reverse_xai_pipeline.ipynb, and reverse_xai_trajectory_panels.ipynb, each
with its own cache file. One implementation here, reused by acceptance
analysis, the surrogate, and XAI alike.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


class ResNetEncoder(nn.Module):
    """ResNet-50 feature extractor (2048-d, pre-avgpool-squeeze output).

    Deliberately keeps each stage as a *named* attribute (conv1, bn1, ...,
    layer1..layer4, avgpool) instead of flattening into a plain
    nn.Sequential indexed by position. The original codebase's
    `nn.Sequential(*list(backbone.children())[:-1])` pattern made "layer4"
    accessible only as `body[7]`; a since-identified bug in the original
    ProxyModel referenced `body[6]` (verified: index 6 is actually layer3,
    index 7 is layer4) while calling the flag `unfreeze_layer4` and
    documenting it as layer4 -- an off-by-one that silently fine-tuned the
    wrong block. Naming eliminates that whole bug class: `self.layer4` can
    never resolve to the wrong stage.
    """

    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.squeeze(-1).squeeze(-1)


PREPROCESS = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class EmbeddingCache:
    """L2-normalized 2048-d ResNet-50 embeddings, persisted to a single .npz."""

    def __init__(self, cache_path: Path, device: str | None = None):
        self.cache_path = cache_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder: ResNetEncoder | None = None
        self._cache: Dict[str, np.ndarray] = {}
        if cache_path.exists():
            loaded = np.load(cache_path, allow_pickle=True)
            self._cache = {k: loaded[k] for k in loaded.files}

    @property
    def encoder(self) -> ResNetEncoder:
        if self._encoder is None:
            self._encoder = ResNetEncoder().to(self.device).eval()
        return self._encoder

    @torch.no_grad()
    def _embed(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        x = PREPROCESS(img).unsqueeze(0).to(self.device)
        feat = self.encoder(x).cpu().numpy().squeeze()
        return (feat / (np.linalg.norm(feat) + 1e-9)).astype(np.float32)

    @staticmethod
    def _key(path: Path) -> str:
        # np.savez/np.load round-trip a dict of arrays through a zip archive,
        # and zipfile.ZipInfo unconditionally converts os.sep to "/" in the
        # stored member name (real behavior, confirmed against this exact
        # cache file: every one of 6,000 Windows-style backslash keys was
        # silently duplicated under a forward-slash name on save). Using
        # as_posix() here means the in-memory key always matches what's
        # actually stored on disk, on every OS, so a reload is a real cache
        # hit instead of a permanent miss that keeps re-embedding everything
        # and re-appending duplicate entries on every save.
        return Path(path).resolve().as_posix()

    def get(self, path: Path) -> np.ndarray:
        key = self._key(path)
        if key not in self._cache:
            self._cache[key] = self._embed(path)
        return self._cache[key]

    def get_many(self, paths: Iterable[Path]) -> np.ndarray:
        return np.stack([self.get(p) for p in paths])

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.cache_path, **self._cache)

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-9)
        b = b / (np.linalg.norm(b) + 1e-9)
        return float(1.0 - np.dot(a, b))
