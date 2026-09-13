"""Attribution methods for the surrogate: Integrated Gradients + SmoothGrad,
Occlusion, and Guided Grad-CAM.

This is the ONE implementation, matching the higher-fidelity configuration
from reverse_xai_pipeline.ipynb (blurred-input IG baseline, more
SmoothGrad samples/IG steps) rather than reverse_xai_trajectory_panels.ipynb's
(zero baseline, fewer samples, fully-frozen backbone) -- see Part II paper,
sec:xai-results, for why the two diverged in attribution quality. There is
no second, lower-fidelity XAI pipeline in this package to accidentally use
by mistake.

Plain Grad-CAM and LIME (the trajectory-panels notebook's original choices)
are deliberately not offered here. On this task -- a thin curve on a mostly
white background -- both underperformed once run at scale: Grad-CAM's 7x7
layer4 feature map is too coarse to localize a 1-2px-wide curve, and LIME's
felzenszwalb superpixels are large/irregular enough that top-weighted
segments frequently land on background, borders, or axis content rather than
the curve. `guided_gradcam_map` and `occlusion_map` are drop-in replacements
that address each specific failure mode; see their docstrings.

Grad-CAM targets `model.encoder.layer4` by name (see surrogate/model.py and
analysis/embeddings.py docstrings for why "by name" matters here: the
original notebooks located this layer by list index, `body[6]`, which is
actually layer3).
"""

from __future__ import annotations

from typing import Tuple

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from captum.attr import GuidedBackprop, IntegratedGradients, LayerGradCam, NoiseTunnel, Occlusion
from scipy.ndimage import gaussian_filter
from torchvision.transforms.functional import gaussian_blur

from ..config import XaiConfig
from ..analysis.embeddings import PREPROCESS
from ..surrogate.model import ProxyModel


def normalize_map(a: np.ndarray, percentile: float = 99) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a, dtype=np.float32))
    hi = np.percentile(a, percentile)
    lo = np.percentile(a, 1)
    if hi <= lo:
        hi, lo = a.max() + 1e-9, a.min()
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def overlay_heatmap_on_image(img: Image.Image, heatmap: np.ndarray, cmap: str = "jet", alpha: float = 0.55) -> np.ndarray:
    """Alpha-blend an already-normalized [0, 1] heatmap over the source image.

    Used by the trajectory-panel view (unlike the single-image xai panels,
    which show the heatmap on its own) so the underlying curve stays visible
    under the attribution, matching reverse_xai_trajectory_panels.ipynb.
    """
    img_arr = np.asarray(img.resize((heatmap.shape[1], heatmap.shape[0]))).astype(np.float32) / 255.0
    heat = matplotlib.colormaps[cmap](np.clip(heatmap, 0, 1))[..., :3]
    return np.uint8(np.clip(img_arr * (1 - alpha) + heat * alpha, 0, 1) * 255)


def _load_and_preprocess(img_path, device: str) -> Tuple[Image.Image, torch.Tensor]:
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    x = PREPROCESS(img).unsqueeze(0).to(device)
    return img, x


def saliency_map(model: ProxyModel, img_path, cfg: XaiConfig, device: str | None = None) -> Tuple[Image.Image, np.ndarray]:
    """Integrated Gradients + SmoothGrad attribution, normalized to [0, 1]."""
    device = device or next(model.parameters()).device
    model.eval()
    img, x = _load_and_preprocess(img_path, device)
    x.requires_grad_(True)

    if cfg.ig_baseline == "blurred":
        baseline = gaussian_blur(x.detach(), kernel_size=cfg.ig_blur_kernel_size, sigma=cfg.ig_blur_sigma)
    elif cfg.ig_baseline == "zero":
        baseline = torch.zeros_like(x)
    else:
        raise ValueError(f"Unknown ig_baseline: {cfg.ig_baseline!r}")

    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr = nt.attribute(
        x,
        baselines=baseline,
        nt_type="smoothgrad",
        nt_samples=cfg.ig_smoothgrad_samples,
        stdevs=0.05,
        n_steps=cfg.ig_steps,
        internal_batch_size=cfg.ig_smoothgrad_samples,
    )
    a = np.abs(attr.squeeze().detach().cpu().numpy()).sum(axis=0)
    a = gaussian_filter(a, sigma=cfg.saliency_smoothing_sigma)
    return img, normalize_map(a)


def gradcam_map(model: ProxyModel, img_path, cfg: XaiConfig, device: str | None = None) -> Tuple[Image.Image, np.ndarray]:
    device = device or next(model.parameters()).device
    model.eval()
    img, x = _load_and_preprocess(img_path, device)

    layer_gc = LayerGradCam(model, model.encoder.layer4)
    attr = layer_gc.attribute(x, relu_attributions=True)
    a = attr.squeeze().detach().cpu()
    if a.ndim == 3:
        a = a.mean(dim=0)
    a_up = F.interpolate(a.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False).squeeze().numpy()
    a_up = gaussian_filter(a_up, sigma=cfg.gradcam_smoothing_sigma)
    return img, normalize_map(a_up)


def guided_gradcam_map(model: ProxyModel, img_path, cfg: XaiConfig, device: str | None = None) -> Tuple[Image.Image, np.ndarray]:
    """Guided Grad-CAM (Selvaraju et al. 2017): the elementwise product of
    Grad-CAM's upsampled layer4 activation map with Guided Backprop's
    pixel-level gradient map.

    Plain Grad-CAM's 7x7 feature map, bicubic-upsampled to 224x224, cannot
    localize a 1-2px-wide curve -- the result is a broad blob regardless of
    what the network actually attends to. Guided Backprop alone has
    pixel-level detail but is not output-discriminative (it looks similar
    regardless of which output/class it's computed for). Multiplying the two
    keeps Grad-CAM's localization (what region matters for *this* output)
    while recovering the fine detail Grad-CAM's low resolution destroys.
    """
    device = device or next(model.parameters()).device
    model.eval()
    img, x = _load_and_preprocess(img_path, device)

    layer_gc = LayerGradCam(model, model.encoder.layer4)
    cam = layer_gc.attribute(x, relu_attributions=True).squeeze().detach().cpu()
    if cam.ndim == 3:
        cam = cam.mean(dim=0)
    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False
    ).squeeze().numpy()
    cam_up = np.clip(cam_up, 0, None)

    guided = GuidedBackprop(model).attribute(x).squeeze().detach().cpu().numpy()
    guided = np.abs(guided).sum(axis=0)

    combined = gaussian_filter(cam_up * guided, sigma=cfg.gradcam_smoothing_sigma)
    return img, normalize_map(combined)


def occlusion_map(model: ProxyModel, img_path, cfg: XaiConfig, device: str | None = None) -> Tuple[Image.Image, np.ndarray]:
    """Sliding-window occlusion attribution (Zeiler & Fergus, 2014), via
    Captum. A deterministic, full-coverage alternative to LIME: LIME's
    felzenszwalb superpixels are large and irregular, and on a thin curve
    over a mostly-white background its top-weighted segments frequently land
    on background, borders, or axis content rather than the curve.
    Occlusion's small, regular, overlapping patches give much finer and more
    curve-localized attribution at comparable cost (~729 forward passes at
    the default patch_size=16/stride=8 vs. LIME's 600 samples)."""
    device = device or next(model.parameters()).device
    model.eval()
    img, x = _load_and_preprocess(img_path, device)

    occlusion = Occlusion(model)
    attr = occlusion.attribute(
        x,
        strides=(3, cfg.occlusion_stride, cfg.occlusion_stride),
        sliding_window_shapes=(3, cfg.occlusion_patch_size, cfg.occlusion_patch_size),
        baselines=0.0,
        perturbations_per_eval=8,  # ~3.5x faster than the default 1, same result (pure batching, no fidelity change)
    )
    a = np.abs(attr.squeeze().detach().cpu().numpy()).sum(axis=0)
    a = gaussian_filter(a, sigma=cfg.saliency_smoothing_sigma)
    return img, normalize_map(a)
