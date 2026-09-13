"""Single CLI entrypoint replacing the original's "open the right notebook
cell / run the right one-off script" workflow.

    python -m kinematics_grading generate  --out generated/ --optuna-dir optuna_logs/
    python -m kinematics_grading grade     --generated generated/ --out grading_results/
    python -m kinematics_grading grade     --generated generated/ --out grading_results/ --retry-failed
    python -m kinematics_grading grade     --generated generated/ --out grading_results/ --retry-until-clean
    python -m kinematics_grading analyze   --grading-results grading_results/ --out analysis_cache/
    python -m kinematics_grading train-surrogate --grading-results grading_results/ --out surrogate_out/
    python -m kinematics_grading xai       --grading-results grading_results/ --model surrogate_out/proxy_model.pt --out xai_out/
    python -m kinematics_grading trajectory-panels --grading-results grading_results/ --model surrogate_out/proxy_model.pt --out trajectory_panels/
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .config import DATA_DIR, load_azure_config, load_settings
from .domain import is_gradable_step


def _add_generate_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("generate", help="Generate the reverse-perturbation corpus.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--optuna-dir", type=Path, required=True)
    p.add_argument("--exercises", type=int, nargs="*", default=None)
    p.add_argument("--config", type=Path, default=None)


def _add_grade_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("grade", help="Grade generated images via the configured Azure OpenAI deployment (async, resumable).")
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exercises", type=int, nargs="*", default=None)
    p.add_argument("--retry-failed", action="store_true", help="Retry outstanding failures once.")
    p.add_argument(
        "--retry-until-clean", action="store_true",
        help="Repeatedly retry outstanding failures (handles a recurring transient network/DNS "
             "outage without needing a fresh manual --retry-failed each time); stops when nothing "
             "is left, a round makes no progress, or --max-retry-rounds is hit.",
    )
    p.add_argument("--max-retry-rounds", type=int, default=20)
    p.add_argument("--retry-pause-seconds", type=float, default=20.0)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="Grade at most N images (smoke testing).")
    p.add_argument(
        "--partial-credit-policy", choices=["step_calibrated", "standalone_submission"], default="step_calibrated",
        help="'step_calibrated' (default): the original policy used by every persisted result so far -- "
             "frames each image as an intermediate correction step and permits early steps partial credit "
             "just for moving in the right direction. 'standalone_submission': a second, independent "
             "experiment -- grades each image as if it were a complete standalone answer (no reference "
             "image shown, no step/expected-progress hint), with partial credit justified strictly by "
             "closeness to each rule's own quantity/relation/expected threshold (see grading/prompts.py). "
             "IMPORTANT: point --out at a fresh directory when using a non-default policy -- grading is "
             "resumable by image path only, so reusing a directory graded under a different policy will "
             "just skip every image as 'already graded' instead of re-grading under the new policy.",
    )


def _add_analyze_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("analyze", help="Acceptance-point + confusing/robust-to-noise analysis.")
    p.add_argument("--grading-results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)


def _add_train_surrogate_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train-surrogate", help="Train the differentiable surrogate on real graded scores.")
    p.add_argument("--grading-results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--embeddings-cache", type=Path, default=None,
        help="Reuse an existing embeddings_cache.npz (e.g. from `analyze`) instead of recomputing.",
    )
    p.add_argument("--config", type=Path, default=None)


def _add_xai_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("xai", help="Saliency + Grad-CAM attribution panels using a trained surrogate.")
    p.add_argument("--grading-results", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="Path to a proxy_model.pt from train-surrogate.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-per-exercise", type=int, default=4, help="Images sampled across the score range, per exercise.")
    p.add_argument("--config", type=Path, default=None)


def _add_trajectory_panels_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "trajectory-panels",
        help="Combined per-trajectory panels (5 graded steps x original/IG+SmoothGrad/"
             "Occlusion/Guided-Grad-CAM) for representative trajectories, adapted from "
             "reverse_xai_trajectory_panels.ipynb.",
    )
    p.add_argument("--grading-results", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="Path to a proxy_model.pt from train-surrogate.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--embeddings-cache", type=Path, default=None,
        help="Reuse an existing embeddings_cache.npz (e.g. from `analyze`) instead of recomputing.",
    )
    p.add_argument(
        "--all-trajectories", action="store_true",
        help="Render every trajectory in the corpus instead of the curated "
             "xai.trajectories_per_pair sample -- includes trajectories that were "
             "never accepted (no green-highlighted step). Expensive: dominated by "
             "IG+SmoothGrad/Occlusion/Guided-Grad-CAM over 5 graded steps per "
             "trajectory; measure per-trajectory time on a small run before committing "
             "to the full corpus.",
    )
    p.add_argument("--config", type=Path, default=None)


def cmd_generate(args: argparse.Namespace) -> int:
    from .generation.pipeline import generate_for_target

    settings = load_settings(args.config)
    exercises = args.exercises or settings.generation.exercises
    for ex in exercises:
        logging.info("=== Generating exercise %s ===", ex)
        generate_for_target(ex, settings.generation, args.out, args.optuna_dir)
    return 0


def cmd_grade(args: argparse.Namespace) -> int:
    from .grading.pipeline import discover_gradable_images, run_grading, run_retry_failed, run_retry_until_clean

    settings = load_settings(args.config)
    azure_cfg = load_azure_config(args.env_file)

    if args.retry_until_clean:
        counters = asyncio.run(
            run_retry_until_clean(
                azure_cfg, args.out, DATA_DIR, args.max_retry_rounds, args.retry_pause_seconds,
                partial_credit_policy=args.partial_credit_policy,
            )
        )
    elif args.retry_failed:
        counters = asyncio.run(
            run_retry_failed(azure_cfg, args.out, DATA_DIR, partial_credit_policy=args.partial_credit_policy)
        )
    else:
        exercises = args.exercises or settings.generation.exercises
        images = discover_gradable_images(args.generated, exercises)
        if args.limit:
            images = images[: args.limit]
        logging.info("Discovered %d gradable images", len(images))
        counters = asyncio.run(
            run_grading(images, azure_cfg, args.out, DATA_DIR, partial_credit_policy=args.partial_credit_policy)
        )

    logging.info("Done: %s", counters)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    import pandas as pd

    from .analysis.acceptance import (
        acceptance_summary_by_exercise,
        first_accepted_step_distribution,
        partial_credit_on_unsatisfied_rules,
        per_step_pass_rate,
        trajectory_summary,
    )
    from .analysis.classification import classify_confusing_robust, confusing_robust_summary
    from .analysis.embeddings import EmbeddingCache
    from .config import DATA_DIR as data_dir
    from .domain import parse_step_info

    settings = load_settings(args.config)
    jsonl_path = args.grading_results / "intermediate_step_rule_grades.jsonl"
    if not jsonl_path.exists():
        logging.error("No grading results found at %s", jsonl_path)
        return 1

    df = pd.read_json(jsonl_path, lines=True)
    df["step_order"] = df["step_file"].apply(lambda s: parse_step_info(s).order if parse_step_info(s) else None)
    df = df.dropna(subset=["step_order"])

    args.out.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(args.out / "embeddings_cache.npz")
    dists = []
    for ex, group in df.groupby("exercise"):
        ref_path = data_dir / f"exercice_{int(ex)}" / f"correct_{int(ex)}.png"
        ref_emb = cache.get(ref_path)
        for _, row in group.iterrows():
            emb = cache.get(Path(row["generated_image"]))
            dists.append(EmbeddingCache.cosine_distance(emb, ref_emb))
    df["dist_to_ref"] = dists
    cache.save()

    traj_df = trajectory_summary(df, settings.analysis.score_threshold)
    traj_df.to_csv(args.out / "trajectory_summary.csv", index=False)

    acc_summary = acceptance_summary_by_exercise(traj_df)
    acc_summary.to_csv(args.out / "acceptance_summary_by_exercise.csv", index=False)

    step_pass_rate = per_step_pass_rate(df, settings.analysis.score_threshold)
    step_pass_rate.to_csv(args.out / "step_pass_rate_by_exercise.csv", index=False)

    first_accept_dist = first_accepted_step_distribution(traj_df)
    first_accept_dist.to_csv(args.out / "first_accepted_step_distribution.csv", index=False)

    partial_credit = partial_credit_on_unsatisfied_rules(df, settings.analysis.score_threshold)
    partial_credit.to_csv(args.out / "partial_credit_on_unsatisfied_rules.csv", index=False)
    partial_credit_early = partial_credit_on_unsatisfied_rules(df[df["step_order"] <= 2], settings.analysis.score_threshold)
    partial_credit_early.to_csv(args.out / "partial_credit_on_unsatisfied_rules_steps_1_2.csv", index=False)

    classified = classify_confusing_robust(df, settings.analysis.embedding_distance_threshold, settings.analysis.score_threshold)
    cr_summary = confusing_robust_summary(classified)
    cr_summary.to_csv(args.out / "confusing_robust_summary.csv", index=False)

    logging.info("Acceptance summary:\n%s", acc_summary.to_string(index=False))
    logging.info("Per-step pass rate by exercise:\n%s", step_pass_rate.to_string(index=False))
    logging.info("First-accepted-step distribution:\n%s", first_accept_dist.to_string(index=False))
    logging.info("Partial credit from unsatisfied rules (all passes):\n%s", partial_credit.to_string(index=False))
    logging.info("Partial credit from unsatisfied rules (steps 1-2 only):\n%s", partial_credit_early.to_string(index=False))
    logging.info("Confusing/robust-to-noise summary:\n%s", cr_summary.to_string(index=False))
    return 0


def cmd_train_surrogate(args: argparse.Namespace) -> int:
    import dataclasses

    import numpy as np
    import pandas as pd
    import torch

    from .analysis.embeddings import EmbeddingCache
    from .surrogate.train import train_proxy

    settings = load_settings(args.config)
    jsonl_path = args.grading_results / "intermediate_step_rule_grades.jsonl"
    if not jsonl_path.exists():
        logging.error("No grading results found at %s", jsonl_path)
        return 1

    df = pd.read_json(jsonl_path, lines=True)
    args.out.mkdir(parents=True, exist_ok=True)

    cache_path = args.embeddings_cache or (args.out / "embeddings_cache.npz")
    cache = EmbeddingCache(cache_path)
    logging.info("Embedding %d images (cached ones are instant)...", len(df))
    embeddings = np.stack([cache.get(Path(p)) for p in df["generated_image"]])
    cache.save()

    scores = df["score_ratio"].to_numpy()
    logging.info(
        "Training surrogate: unfreeze_layer=%s, unfreeze_after_epoch=%d, total_epochs=%d",
        settings.surrogate.unfreeze_layer, settings.surrogate.unfreeze_after_epoch, settings.surrogate.total_epochs,
    )
    model, history = train_proxy(embeddings, scores, settings.surrogate)

    torch.save(model.state_dict(), args.out / "proxy_model.pt")
    pd.DataFrame([dataclasses.asdict(h) for h in history]).to_csv(args.out / "proxy_training_log.csv", index=False)

    final = history[-1]
    logging.info(
        "Surrogate trained: %d epochs | final val MAE=%.4f RMSE=%.4f acc(+/-0.15)=%.1f%% "
        "Pearson r=%.3f Spearman=%.3f",
        len(history), final.mae, final.rmse, final.accuracy_within_0_15, final.pearson_r, final.spearman_r,
    )
    return 0


def cmd_xai(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from .surrogate.model import ProxyModel
    from .xai.attributions import gradcam_map, saliency_map

    settings = load_settings(args.config)
    jsonl_path = args.grading_results / "intermediate_step_rule_grades.jsonl"
    if not jsonl_path.exists():
        logging.error("No grading results found at %s", jsonl_path)
        return 1
    if not args.model.exists():
        logging.error("No trained surrogate found at %s (run train-surrogate first)", args.model)
        return 1

    df = pd.read_json(jsonl_path, lines=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ProxyModel().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)

    for ex, group in df.groupby("exercise"):
        group = group.sort_values("score_ratio").reset_index(drop=True)
        n = min(args.n_per_exercise, len(group))
        sample_idx = np.linspace(0, len(group) - 1, n).astype(int)

        for idx in sample_idx:
            row = group.iloc[int(idx)]
            img_path = Path(row["generated_image"])
            _, sal = saliency_map(model, img_path, settings.xai, device)
            _, cam = gradcam_map(model, img_path, settings.xai, device)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(Image.open(img_path).convert("RGB").resize((224, 224)))
            axes[0].set_title(f"ex{int(ex)} score={row['score_ratio']:.2f}")
            axes[1].imshow(sal, cmap="jet")
            axes[1].set_title("IG + SmoothGrad")
            axes[2].imshow(cam, cmap="jet")
            axes[2].set_title("Grad-CAM")
            for ax in axes:
                ax.axis("off")

            out_path = args.out / f"ex{int(ex)}_{img_path.parent.name}_{img_path.stem}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logging.info("Saved %s (score=%.2f)", out_path, row["score_ratio"])

    return 0


def _style_panel_axis(ax, highlighted: bool) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        if highlighted:
            spine.set_visible(True)
            spine.set_linewidth(4)
            spine.set_edgecolor("limegreen")
        else:
            spine.set_visible(False)


def cmd_trajectory_panels(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from .analysis.acceptance import trajectory_summary
    from .analysis.embeddings import EmbeddingCache
    from .domain import parse_step_info
    from .surrogate.model import ProxyModel
    from .xai.attributions import guided_gradcam_map, occlusion_map, overlay_heatmap_on_image, saliency_map
    from .xai.trajectory_panels import build_panel_items, select_representative_trajectories

    settings = load_settings(args.config)
    jsonl_path = args.grading_results / "intermediate_step_rule_grades.jsonl"
    if not jsonl_path.exists():
        logging.error("No grading results found at %s", jsonl_path)
        return 1
    if not args.model.exists():
        logging.error("No trained surrogate found at %s (run train-surrogate first)", args.model)
        return 1

    df = pd.read_json(jsonl_path, lines=True)
    df["step_order"] = df["step_file"].apply(lambda s: parse_step_info(s).order if parse_step_info(s) else None)
    df = df.dropna(subset=["step_order"])

    args.out.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(args.embeddings_cache or (args.out / "embeddings_cache.npz"))
    dists = []
    for ex, group in df.groupby("exercise"):
        ref_path = DATA_DIR / f"exercice_{int(ex)}" / f"correct_{int(ex)}.png"
        ref_emb = cache.get(ref_path)
        for _, row in group.iterrows():
            emb = cache.get(Path(row["generated_image"]))
            dists.append(EmbeddingCache.cosine_distance(emb, ref_emb))
    df["dist_to_ref"] = dists
    cache.save()

    traj_df = trajectory_summary(df, settings.grading.score_threshold)
    if args.all_trajectories:
        selected = traj_df.copy()
        logging.info(
            "Rendering all %d trajectories (%d accepted, %d never accepted)",
            len(selected), int(traj_df["has_acceptance"].sum()), int((~traj_df["has_acceptance"]).sum()),
        )
    else:
        selected = select_representative_trajectories(traj_df, settings.xai.trajectories_per_pair)
        logging.info(
            "Selected %d representative trajectories (%d accepted of %d total)",
            len(selected), int(traj_df["has_acceptance"].sum()), len(traj_df),
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ProxyModel().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    methods = ["original", "saliency", "occlusion", "guided_gradcam"]
    row_titles = {
        "original": "Original", "saliency": "IG + SmoothGrad",
        "occlusion": "Occlusion", "guided_gradcam": "Guided Grad-CAM",
    }

    saved = 0
    skipped = 0
    for _, traj_row in selected.iterrows():
        out_path = (
            args.out
            / f"ex{int(traj_row['exercise'])}_from{int(traj_row['source_exercise'])}"
              f"_r{int(traj_row['round'])}_{traj_row['instance']}.png"
        )
        if out_path.exists():
            skipped += 1
            continue

        group = df[
            (df["exercise"] == traj_row["exercise"])
            & (df["round"] == traj_row["round"])
            & (df["source_exercise"] == traj_row["source_exercise"])
            & (df["instance"] == traj_row["instance"])
        ]
        items, first_accepted = build_panel_items(group, settings.grading.score_threshold)
        if not items:
            logging.warning("No panel items resolved for trajectory %s, skipping", dict(traj_row))
            continue

        fig, axes = plt.subplots(len(methods), len(items), figsize=(3.2 * len(items), 13))
        if len(items) == 1:
            axes = axes.reshape(len(methods), 1)

        for r, method in enumerate(methods):
            axes[r, 0].set_ylabel(row_titles[method], fontsize=12, fontweight="bold")
            for c, item in enumerate(items):
                ax = axes[r, c]

                if method == "original":
                    arr = np.asarray(Image.open(item.path).convert("RGB").resize((224, 224)))
                elif method == "saliency":
                    img, sal = saliency_map(model, item.path, settings.xai, device)
                    arr = overlay_heatmap_on_image(img, sal)
                elif method == "occlusion":
                    img, occ = occlusion_map(model, item.path, settings.xai, device)
                    arr = overlay_heatmap_on_image(img, occ)
                else:  # guided_gradcam
                    img, gcam = guided_gradcam_map(model, item.path, settings.xai, device)
                    arr = overlay_heatmap_on_image(img, gcam)

                ax.imshow(arr)
                _style_panel_axis(ax, highlighted=item.is_first_accepted)

                if r == 0:
                    ax.set_title(item.label, fontsize=10)
                    text = f"score={item.score_text}"
                    if item.is_first_accepted:
                        text += f"\nFIRST >= {settings.grading.score_threshold:.2f}"
                    ax.text(
                        0.5, -0.08, text, transform=ax.transAxes, ha="center", va="top", fontsize=9,
                        color="green" if item.is_first_accepted else "black",
                        fontweight="bold" if item.is_first_accepted else "normal",
                    )

        title = (
            f"Exercise {int(traj_row['exercise'])} | from_correct_{int(traj_row['source_exercise'])} | "
            f"round {int(traj_row['round'])} | {traj_row['instance']} | first accepted step={first_accepted}"
        )
        fig.suptitle(title, fontsize=16, y=0.995)
        fig.tight_layout()

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved += 1
        logging.info("Saved %s (first accepted step=%s)", out_path, first_accepted)

    logging.info("Saved %d trajectory panels (%d already present, skipped)", saved, skipped)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(prog="kinematics-grading")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_generate_parser(sub)
    _add_grade_parser(sub)
    _add_analyze_parser(sub)
    _add_train_surrogate_parser(sub)
    _add_xai_parser(sub)
    _add_trajectory_panels_parser(sub)

    args = parser.parse_args(argv)
    handlers = {
        "generate": cmd_generate,
        "grade": cmd_grade,
        "analyze": cmd_analyze,
        "train-surrogate": cmd_train_surrogate,
        "xai": cmd_xai,
        "trajectory-panels": cmd_trajectory_panels,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
