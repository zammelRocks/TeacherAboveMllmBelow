"""One surrogate model implementation, with configurable staged fine-tuning.

The original codebase had two divergent proxy models: reverse_xai_pipeline.ipynb's
(labeled "progressive layer4 unfreezing", r=0.889) and
reverse_xai_trajectory_panels.ipynb's (fully frozen backbone, r~0.87-0.88 but
with visibly worse Grad-CAM/IG localization -- see Part II paper,
sec:xai-results). This class supports both configurations through
`unfreeze_layer`, defaulting to the good one, so a caller has to opt out of
fine-tuning rather than accidentally omit it.

Note: the original "layer4" pipeline actually unfroze `body[6]`, which is
layer3, not layer4 (body[7]) -- verified directly against torchvision's
child ordering. `set_backbone_trainable` here addresses layers by name via
ResNetEncoder's named attributes, so it fine-tunes the layer it's told to,
not whichever one happened to sit at a given list index.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..analysis.embeddings import ResNetEncoder


class ProxyModel(nn.Module):
    def __init__(self, encoder: Optional[ResNetEncoder] = None):
        super().__init__()
        self.encoder = encoder or ResNetEncoder()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.head(feat).squeeze(1)

    def predict_from_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        return self.head(emb).squeeze(1)

    def set_backbone_trainable(self, unfreeze_layer: Optional[str]) -> None:
        """unfreeze_layer=None keeps the whole backbone frozen (fast, lower
        fidelity, matches the trajectory-panels pipeline). unfreeze_layer=
        'layer4' (recommended default; see config/default.yaml) unfreezes
        exactly that named block for fine-tuning -- addressed by attribute
        name on ResNetEncoder, so there is no positional index to get wrong."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        if unfreeze_layer is not None:
            target = getattr(self.encoder, unfreeze_layer)
            for p in target.parameters():
                p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
