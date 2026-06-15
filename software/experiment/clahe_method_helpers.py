#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared CLAHE method helpers for public diagnostic scripts.

The public artifact intentionally does not bundle raw datasets, raw videos, or
student checkpoint weights. The fixed and rule-based CLAHE paths work directly;
the DT-QAT student path is available when the user supplies the local weight
file listed by ``STUDENT_WEIGHTS``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parents[1]
DISTILL_OUT = ARTIFACT_ROOT / "software" / "distillation" / "distill_out"

STUDENT_WEIGHTS = DISTILL_OUT / "student_o5_multihead.pt"
STUDENT_NORM = DISTILL_OUT / "obs_norm_o5_multihead.json"

FIXED_CL_BASE = 2.0
CL_MIN = 0.1
CL_MAX = 20.0
INIT_CL = 5.0
MAX_STEPS = 5
DELTA_CL_CLIP = 2.0
DELTA_CL_MAX = 2.0


def calc_renyi_entropy(image: np.ndarray) -> float:
    hist = np.histogram(image, bins=256, range=(0, 256))[0].astype(np.float64)
    p = hist / max(float(image.size), 1.0)
    s = float(np.sum(p * p))
    return 0.0 if s <= 1e-12 else float(-math.log2(s))


def global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mux = float(x.mean())
    muy = float(y.mean())
    varx = float(((x - mux) ** 2).mean())
    vary = float(((y - muy) ** 2).mean())
    cov = float(((x - mux) * (y - muy)).mean())
    denom = (mux * mux + muy * muy + c1) * (varx + vary + c2)
    return 1.0 if denom == 0 else ((2 * mux * muy + c1) * (2 * cov + c2)) / denom


def apply_clahe(image: np.ndarray, cl: float) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=float(cl), tileGridSize=(8, 8)).apply(image)


def compute_rule_based_cl(image: np.ndarray) -> float:
    mu = float(np.mean(image))
    var = float(np.var(image))
    h2 = calc_renyi_entropy(image)
    norm_mean = mu / 255.0
    norm_var = min(var / 5000.0, 1.0)
    norm_h2 = float(np.clip((h2 - 4.0) / 4.0, 0.0, 1.0))
    score = (1.0 - norm_mean) * 0.5 + (1.0 - norm_var) * 0.3 + (1.0 - norm_h2) * 0.2
    return float(np.clip(2.0 + score * 12.0, CL_MIN, CL_MAX))


def apply_rule_based(image: np.ndarray) -> tuple[np.ndarray, float]:
    cl = compute_rule_based_cl(image)
    return apply_clahe(image, cl), cl


class MLP5:
    def __init__(self, weights: Path, norm: Path):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.trunk = nn.Sequential(
                    nn.Linear(5, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                )
                self.head_a = nn.Linear(64, 1)
                self.head_aux = nn.Linear(64, 1)

            def forward(self, x: Any) -> tuple[Any, Any]:
                z = self.trunk(x)
                return self.head_a(z), self.head_aux(z)

        if not weights.is_file() or not norm.is_file():
            raise FileNotFoundError(
                "Missing DT-QAT student assets. The public artifact does not "
                f"bundle the checkpoint by default: weights={weights}, norm={norm}"
            )
        self.torch = torch
        self.net = Net()
        self.net.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
        self.net.eval()
        norm_json = json.loads(norm.read_text(encoding="utf-8"))
        self.mean = np.asarray(norm_json["mean"], dtype=np.float32)
        self.std = np.asarray(norm_json["std"], dtype=np.float32)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        self.delta_cl_max = float(norm_json.get("delta_cl_max", DELTA_CL_MAX))

    def infer_cl(self, image: np.ndarray) -> float:
        mu = float(np.mean(image))
        var = float(np.var(image))
        h2 = calc_renyi_entropy(image)
        cl = float(INIT_CL)
        for t in range(MAX_STEPS):
            x = np.asarray([mu, var, h2, cl, float(t)], dtype=np.float32)
            xn = (x - self.mean) / self.std
            xt = self.torch.from_numpy(xn).unsqueeze(0)
            with self.torch.no_grad():
                y_raw, _ = self.net(xt)
                y_raw_f = float(y_raw[0, 0].item())
            delta = math.tanh(y_raw_f) * self.delta_cl_max
            delta = float(np.clip(delta, -DELTA_CL_CLIP, DELTA_CL_CLIP))
            cl = float(np.clip(cl + delta, CL_MIN, CL_MAX))
        return cl


def apply_policy_clahe(image: np.ndarray, policy: Any) -> tuple[np.ndarray, float]:
    cl = policy.infer_cl(image)
    return apply_clahe(image, cl), cl
