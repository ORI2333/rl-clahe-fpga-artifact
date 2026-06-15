#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask-aware industrial defect diagnostic for the TECS revision.

Typical commands from the repository root:

PowerShell, MVTec AD layout:
  & "E:\\Program Files\\Anaconda\\envs\\myenv312\\python.exe" `
    ".\\project\\RL_CLAHE\\analysis\\masked_industrial_compare\\masked_industrial_compare.py" `
    --mvtec-root ".\\project\\datasets\\mvtec_anomaly_detection" `
    --categories carpet grid leather tile wood metal_nut hazelnut cable `
    --output-root ".\\project\\RL_CLAHE\\analysis\\masked_industrial_compare\\out\\mvtec_subset"

PowerShell, custom CSV with image_path,mask_path,class,defect columns:
  & "E:\\Program Files\\Anaconda\\envs\\myenv312\\python.exe" `
    ".\\project\\RL_CLAHE\\analysis\\masked_industrial_compare\\masked_industrial_compare.py" `
    --pairs-csv ".\\project\\datasets\\custom_masked_pairs.csv" `
    --output-root ".\\project\\RL_CLAHE\\analysis\\masked_industrial_compare\\out\\custom_masked"

Boundary:
  This is not a trained industrial detector. It uses ground-truth defect masks
  to evaluate defect/background separability and simple saliency-map behavior
  after enhancement. Report it as mask-aware diagnostic evidence, not as a
  detector-level deployment benchmark.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

sys.path.insert(0, str(SCRIPT_DIR))
from clahe_method_helpers import (  # noqa: E402
    FIXED_CL_BASE,
    STUDENT_NORM,
    STUDENT_WEIGHTS,
    MLP5,
    apply_clahe,
    apply_policy_clahe,
    apply_rule_based,
    global_ssim,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_OUT_ROOT = SCRIPT_DIR / "out" / "masked_industrial"


def read_gray(path: Path, resize: tuple[int, int] | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def read_mask(path: Path, resize: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    if resize is not None:
        mask = cv2.resize(mask, resize, interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def find_mvtec_mask(category_dir: Path, defect: str, image_path: Path) -> Path | None:
    candidates = [
        category_dir / "ground_truth" / defect / f"{image_path.stem}_mask.png",
        category_dir / "ground_truth" / defect / f"{image_path.stem}.png",
        category_dir / "ground_truth" / defect / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    gt_dir = category_dir / "ground_truth" / defect
    if gt_dir.is_dir():
        matches = sorted(gt_dir.glob(f"{image_path.stem}*"))
        for match in matches:
            if match.suffix.lower() in IMAGE_EXTS:
                return match
    return None


def discover_mvtec(root: Path, categories: list[str], max_per_defect: int) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise SystemExit(f"MVTec root not found: {root}")
    category_dirs = [root / c for c in categories] if categories else [p for p in sorted(root.iterdir()) if p.is_dir()]
    items: list[dict[str, Any]] = []
    for category_dir in category_dirs:
        test_dir = category_dir / "test"
        if not test_dir.is_dir():
            continue
        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir() and p.name != "good"):
            images = sorted(p for p in defect_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if max_per_defect > 0:
                images = images[:max_per_defect]
            for image_path in images:
                mask_path = find_mvtec_mask(category_dir, defect_dir.name, image_path)
                if mask_path is None:
                    continue
                items.append(
                    {
                        "class": category_dir.name,
                        "defect": defect_dir.name,
                        "image_path": image_path,
                        "mask_path": mask_path,
                    }
                )
    if not items:
        raise SystemExit(f"No masked MVTec defect images found under {root}")
    return items


def discover_pairs(csv_path: Path, max_items: int) -> list[dict[str, Any]]:
    base = csv_path.parent
    items: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"])
            if not image_path.is_absolute():
                image_path = (base / image_path).resolve()
            if not mask_path.is_absolute():
                mask_path = (base / mask_path).resolve()
            items.append(
                {
                    "class": row.get("class", "all"),
                    "defect": row.get("defect", "defect"),
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            )
            if max_items > 0 and len(items) >= max_items:
                break
    if not items:
        raise SystemExit(f"No image/mask pairs found in {csv_path}")
    return items


def safe_name(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " ._-":
            out.append("_")
    return "".join(out).strip("_")[:100]


def mean_or_nan(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(statistics.mean(finite)) if finite else float("nan")


def std_or_nan(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) > 1:
        return float(statistics.stdev(finite))
    if len(finite) == 1:
        return 0.0
    return float("nan")


def simple_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = scores.reshape(-1).astype(np.float64)
    labels = labels.reshape(-1).astype(np.uint8)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def best_dice(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = scores.reshape(-1).astype(np.float64)
    labels = labels.reshape(-1).astype(np.uint8)
    if labels.sum() == 0:
        return float("nan")
    thresholds = np.unique(np.percentile(scores, np.linspace(0, 100, 101)))
    best = 0.0
    for threshold in thresholds:
        pred = scores >= threshold
        tp = float(np.logical_and(pred, labels == 1).sum())
        fp = float(np.logical_and(pred, labels == 0).sum())
        fn = float(np.logical_and(~pred, labels == 1).sum())
        denom = 2.0 * tp + fp + fn
        dice = 0.0 if denom == 0 else 2.0 * tp / denom
        best = max(best, dice)
    return best


def high_frequency_energy(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2, sigmaY=1.2)
    hp = img - blur
    return hp * hp


def saliency_map(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    local = cv2.GaussianBlur(img, (0, 0), sigmaX=5.0, sigmaY=5.0)
    contrast = np.abs(img - local)
    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    return contrast + 0.25 * grad


def masked_metrics(out_img: np.ndarray, ref_img: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    defect = mask.astype(bool)
    background = ~defect
    if not defect.any() or not background.any():
        return {}
    out = out_img.astype(np.float32)
    ref = ref_img.astype(np.float32)
    defect_mean = float(out[defect].mean())
    background_mean = float(out[background].mean())
    defect_std = float(out[defect].std())
    background_std = float(out[background].std())
    contrast = abs(defect_mean - background_mean)
    cnr = contrast / math.sqrt(defect_std * defect_std + background_std * background_std + 1e-9)

    ref_defect_mean = float(ref[defect].mean())
    ref_background_mean = float(ref[background].mean())
    ref_defect_std = float(ref[defect].std())
    ref_background_std = float(ref[background].std())
    ref_contrast = abs(ref_defect_mean - ref_background_mean)
    ref_cnr = ref_contrast / math.sqrt(ref_defect_std * ref_defect_std + ref_background_std * ref_background_std + 1e-9)

    out_hf = high_frequency_energy(out_img)
    ref_hf = high_frequency_energy(ref_img)
    background_hf_gain = float(out_hf[background].mean() / max(float(ref_hf[background].mean()), 1e-9))
    defect_hf_gain = float(out_hf[defect].mean() / max(float(ref_hf[defect].mean()), 1e-9))
    sal = saliency_map(out_img)
    return {
        "defect_mean": defect_mean,
        "background_mean": background_mean,
        "defect_background_contrast": contrast,
        "defect_background_contrast_gain": contrast / max(ref_contrast, 1e-9),
        "cnr": cnr,
        "cnr_gain": cnr / max(ref_cnr, 1e-9),
        "background_hf_gain": background_hf_gain,
        "defect_hf_gain": defect_hf_gain,
        "hf_selectivity": defect_hf_gain / max(background_hf_gain, 1e-9),
        "saliency_auc": simple_auc(sal, mask),
        "saliency_best_dice": best_dice(sal, mask),
        "mae_vs_input": float(np.mean(np.abs(out - ref))),
        "ssim_vs_input": global_ssim(out_img, ref_img),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    metric_fields = [
        "CL",
        "defect_background_contrast",
        "defect_background_contrast_gain",
        "cnr",
        "cnr_gain",
        "background_hf_gain",
        "defect_hf_gain",
        "hf_selectivity",
        "saliency_auc",
        "saliency_best_dice",
        "mae_vs_input",
        "ssim_vs_input",
    ]
    groups: OrderedDict[tuple[str, ...], list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, items in groups.items():
        rec: dict[str, Any] = {field: value for field, value in zip(group_fields, key)}
        rec["n"] = len(items)
        for metric in metric_fields:
            vals = [float(item[metric]) for item in items]
            rec[f"{metric}_mean"] = mean_or_nan(vals)
            rec[f"{metric}_std"] = std_or_nan(vals)
        out.append(rec)
    return out


def fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def save_panel(path: Path, examples: list[dict[str, Any]], method_images: dict[tuple[int, str], np.ndarray]) -> None:
    if not examples:
        return
    methods = ["Input", "Fixed CLAHE (CL=2.0)", "Rule-based Adaptive CLAHE", "Proposed DT-QAT Student"]
    labels = ["Input", "Fixed CL=2.0", "Rule adaptive", "Proposed"]
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    cell = 180
    left = 150
    top = 34
    canvas = Image.new("RGB", (left + len(methods) * cell, top + len(examples) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(labels):
        draw.text((left + col * cell + 8, 9), label, fill=(0, 0, 0), font=font)
    for row_idx, item in enumerate(examples):
        y = top + row_idx * cell
        draw.text((8, y + 8), str(item["class"]), fill=(0, 0, 0), font=font)
        draw.text((8, y + 28), str(item["defect"]), fill=(80, 80, 80), font=small)
        for col, method in enumerate(methods):
            img = method_images.get((int(item["image_index"]), method))
            if img is None:
                continue
            pil = Image.fromarray(img, mode="L").convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)
            x = left + col * cell
            canvas.paste(pil, (x + 10, y + 10))
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(220, 220, 220))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mask-aware industrial defect diagnostic.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--mvtec-root", type=Path)
    src.add_argument("--pairs-csv", type=Path)
    parser.add_argument("--categories", nargs="*", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--max-per-defect", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--resize", default="", help="Optional WIDTHxHEIGHT resize.")
    parser.add_argument(
        "--include-student",
        action="store_true",
        help="Also run the DT-QAT student policy. This requires local student weights that are not bundled with the public artifact.",
    )
    parser.add_argument("--panel-count", type=int, default=8)
    return parser


def parse_resize(value: str) -> tuple[int, int] | None:
    if not value:
        return None
    clean = value.lower().replace("*", "x")
    if "x" not in clean:
        raise argparse.ArgumentTypeError("resize must be WIDTHxHEIGHT")
    w, h = clean.split("x", 1)
    return int(w), int(h)


def write_summary(path: Path, overall: list[dict[str, Any]], by_class: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Mask-Aware Industrial Defect Diagnostic",
        "",
        "## Scope",
        "- Uses ground-truth masks to measure defect/background separability after enhancement.",
        "- Does not train or evaluate a deployed industrial detector.",
        "- Pixel-level saliency AUROC and best Dice are computed from a simple contrast/gradient saliency map, so they are diagnostic proxies, not detector benchmark scores.",
        "",
        "## Overall Summary",
        "| Method | n | CL | Contrast gain | CNR gain | Background HF gain | Defect HF gain | HF selectivity | Saliency AUROC | Best Dice |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall:
        lines.append(
            "| {m} | {n} | {cl} | {cg} | {cnr} | {bhf} | {dhf} | {sel} | {auc} | {dice} |".format(
                m=row["method"],
                n=row["n"],
                cl=fmt(float(row["CL_mean"])),
                cg=fmt(float(row["defect_background_contrast_gain_mean"])),
                cnr=fmt(float(row["cnr_gain_mean"])),
                bhf=fmt(float(row["background_hf_gain_mean"])),
                dhf=fmt(float(row["defect_hf_gain_mean"])),
                sel=fmt(float(row["hf_selectivity_mean"])),
                auc=fmt(float(row["saliency_auc_mean"])),
                dice=fmt(float(row["saliency_best_dice_mean"])),
            )
        )
    lines.extend(["", "## By-Class Summary", "| Class | Method | n | Contrast gain | CNR gain | Saliency AUROC | Best Dice |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in by_class:
        lines.append(
            "| {c} | {m} | {n} | {cg} | {cnr} | {auc} | {dice} |".format(
                c=row["class"],
                m=row["method"],
                n=row["n"],
                cg=fmt(float(row["defect_background_contrast_gain_mean"])),
                cnr=fmt(float(row["cnr_gain_mean"])),
                auc=fmt(float(row["saliency_auc_mean"])),
                dice=fmt(float(row["saliency_best_dice_mean"])),
            )
        )
    lines.extend(["", "## Command Arguments", f"- `{args}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    resize = parse_resize(args.resize)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.mvtec_root:
        items = discover_mvtec(args.mvtec_root.resolve(), args.categories, args.max_per_defect)
        if args.max_items > 0:
            items = items[: args.max_items]
    else:
        items = discover_pairs(args.pairs_csv.resolve(), args.max_items)

    student = MLP5(STUDENT_WEIGHTS, STUDENT_NORM) if args.include_student else None
    methods: OrderedDict[str, Callable[[np.ndarray], tuple[np.ndarray, float]]] = OrderedDict()
    methods["Input"] = lambda img: (img.copy(), float("nan"))
    methods["Fixed CLAHE (CL=2.0)"] = lambda img: (apply_clahe(img, FIXED_CL_BASE), FIXED_CL_BASE)
    methods["Rule-based Adaptive CLAHE"] = apply_rule_based
    if student is not None:
        methods["Proposed DT-QAT Student"] = lambda img, model=student: apply_policy_clahe(img, model)

    rows: list[dict[str, Any]] = []
    panel_items: list[dict[str, Any]] = []
    panel_images: dict[tuple[int, str], np.ndarray] = {}

    for idx, item in enumerate(items):
        image = read_gray(Path(item["image_path"]), resize)
        mask = read_mask(Path(item["mask_path"]), resize)
        if idx < args.panel_count:
            panel_items.append({**item, "image_index": idx})
        for method, func in methods.items():
            out_img, cl = func(image)
            metrics = masked_metrics(out_img, image, mask)
            row = {
                "image_index": idx,
                "class": item["class"],
                "defect": item["defect"],
                "image_path": str(item["image_path"]).replace("\\", "/"),
                "mask_path": str(item["mask_path"]).replace("\\", "/"),
                "method": method,
                "CL": cl,
                **metrics,
            }
            rows.append(row)
            if idx < args.panel_count:
                panel_images[(idx, method)] = out_img

    overall = summarize(rows, ("method",))
    by_class = summarize(rows, ("class", "method"))
    by_defect = summarize(rows, ("class", "defect", "method"))
    write_csv(output_root / "masked_method_detail.csv", rows)
    write_csv(output_root / "masked_summary_overall.csv", overall)
    write_csv(output_root / "masked_summary_by_class.csv", by_class)
    write_csv(output_root / "masked_summary_by_defect.csv", by_defect)
    save_panel(output_root / "masked_representative_panel.png", panel_items, panel_images)
    write_summary(output_root / "analysis_summary.md", overall, by_class, args)

    print(f"items={len(items)}")
    print(f"output_root={output_root}")
    print(f"summary={output_root / 'analysis_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
