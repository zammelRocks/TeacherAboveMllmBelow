"""Single source of truth for pipeline configuration.

Replaces the scattered module-level constants (SCORE_THRESHOLD, N_ROUNDS,
INSTANCES_PER_ROUND, OPTUNA_TRIALS, SIM_THRESHOLD, ...) that were duplicated
with slightly different values/comments across generate.py and the three
grading scripts in the original diffusion_module codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml
from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "default.yaml"
DATA_DIR = PACKAGE_ROOT / "data"


@dataclass(frozen=True)
class GenerationConfig:
    exercises: List[int]
    source_map: Dict[int, List[int]]
    n_rounds: int
    instances_per_round: int
    optuna_trials: int
    image_size: int
    n_curve_points: int


@dataclass(frozen=True)
class GradingConfig:
    score_threshold: float
    intermediate_step_pattern: str
    skip_filenames: List[str]


@dataclass(frozen=True)
class AnalysisConfig:
    embedding_distance_threshold: float
    score_threshold: float


@dataclass(frozen=True)
class SurrogateConfig:
    unfreeze_layer: str
    unfreeze_after_epoch: int
    total_epochs: int
    batch_size: int
    val_fraction: float
    huber_delta: float


@dataclass(frozen=True)
class XaiConfig:
    ig_baseline: str
    ig_smoothgrad_samples: int
    ig_steps: int
    ig_blur_kernel_size: int
    ig_blur_sigma: float
    saliency_smoothing_sigma: int
    gradcam_smoothing_sigma: int
    occlusion_patch_size: int
    occlusion_stride: int
    trajectories_per_pair: int


@dataclass(frozen=True)
class AzureConfig:
    api_key: str
    endpoint: str
    deployment: str
    api_version: str = "2024-02-15-preview"
    max_retries: int = 5
    request_timeout: float = 90.0
    connect_timeout: float = 20.0
    max_concurrency: int = 8
    sleep_between_requests: float = 0.0

    @property
    def chat_url(self) -> str:
        endpoint = self.endpoint.rstrip()
        if "/chat/completions" in endpoint:
            return endpoint
        endpoint = endpoint.rstrip("/")
        return f"{endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"


@dataclass(frozen=True)
class Settings:
    generation: GenerationConfig
    grading: GradingConfig
    analysis: AnalysisConfig
    surrogate: SurrogateConfig
    xai: XaiConfig
    data_dir: Path = field(default=DATA_DIR)


def load_settings(config_path: Path | None = None) -> Settings:
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    gen = raw["generation"]
    grd = raw["grading"]
    ana = raw["analysis"]
    sur = raw["surrogate"]
    xai = raw["xai"]

    return Settings(
        generation=GenerationConfig(
            exercises=list(gen["exercises"]),
            source_map={int(k): list(v) for k, v in gen["source_map"].items()},
            n_rounds=int(gen["n_rounds"]),
            instances_per_round=int(gen["instances_per_round"]),
            optuna_trials=int(gen["optuna_trials"]),
            image_size=int(gen["image_size"]),
            n_curve_points=int(gen["n_curve_points"]),
        ),
        grading=GradingConfig(
            score_threshold=float(grd["score_threshold"]),
            intermediate_step_pattern=str(grd["intermediate_step_pattern"]),
            skip_filenames=list(grd["skip_filenames"]),
        ),
        analysis=AnalysisConfig(
            embedding_distance_threshold=float(ana["embedding_distance_threshold"]),
            score_threshold=float(ana["score_threshold"]),
        ),
        surrogate=SurrogateConfig(
            unfreeze_layer=str(sur["unfreeze_layer"]),
            unfreeze_after_epoch=int(sur["unfreeze_after_epoch"]),
            total_epochs=int(sur["total_epochs"]),
            batch_size=int(sur["batch_size"]),
            val_fraction=float(sur["val_fraction"]),
            huber_delta=float(sur["huber_delta"]),
        ),
        xai=XaiConfig(
            ig_baseline=str(xai["ig_baseline"]),
            ig_smoothgrad_samples=int(xai["ig_smoothgrad_samples"]),
            ig_steps=int(xai["ig_steps"]),
            ig_blur_kernel_size=int(xai["ig_blur_kernel_size"]),
            ig_blur_sigma=float(xai["ig_blur_sigma"]),
            saliency_smoothing_sigma=int(xai["saliency_smoothing_sigma"]),
            gradcam_smoothing_sigma=int(xai["gradcam_smoothing_sigma"]),
            occlusion_patch_size=int(xai["occlusion_patch_size"]),
            occlusion_stride=int(xai["occlusion_stride"]),
            trajectories_per_pair=int(xai["trajectories_per_pair"]),
        ),
    )


def load_azure_config(env_path: Path | None = None, **overrides) -> AzureConfig:
    load_dotenv(dotenv_path=env_path)

    api_key = overrides.get("api_key") or os.getenv("API_KEY")
    endpoint = overrides.get("endpoint") or os.getenv("ENDPOINT")
    deployment = overrides.get("deployment") or os.getenv("DEPLOYMENT")

    if not all([api_key, endpoint, deployment]):
        raise ValueError("Missing Azure OpenAI credentials. Set API_KEY, ENDPOINT, DEPLOYMENT (.env or env vars).")

    return AzureConfig(
        api_key=api_key,
        endpoint=endpoint,
        deployment=deployment,
        api_version=overrides.get("api_version") or os.getenv("API_VERSION", "2024-02-15-preview"),
        max_retries=int(overrides.get("max_retries") or os.getenv("MAX_RETRIES", 5)),
        request_timeout=float(overrides.get("request_timeout") or os.getenv("REQUEST_TIMEOUT", 90)),
        connect_timeout=float(overrides.get("connect_timeout") or os.getenv("CONNECT_TIMEOUT", 20)),
        max_concurrency=int(overrides.get("max_concurrency") or os.getenv("MAX_CONCURRENCY", 8)),
        sleep_between_requests=float(overrides.get("sleep_between_requests") or os.getenv("SLEEP_BETWEEN_REQUESTS", 0.0)),
    )
