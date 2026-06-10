# -*- coding: utf-8 -*-
"""
run_metrics_on_testset_V4.py

8 方法:
1) 原始输入 (Input)
2) 全局 HE
3) 固定参数 CLAHE (CL=2.0)
4) 经验规则自适应 CLAHE
5) 自适应 Gamma 校正
6) MSR Retinex 增强
7) SAC 单教师学生网络   -> 使用 SAC Lite/Rich 教师策略网络
8) 本文：双教师蒸馏 + QAT 学生 -> student_o5_multihead.pt

添加(1) 均值±标准差、(2) 分场景统计、(3) 分布（CDF/箱线图）、(4) 尾部分位数与 worst-10%，

运行位置:
  在 4.SAC/experiment 目录下:
    python run_metrics_on_testset_V3.py
"""

import os
import csv
import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import pyiqa

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from collections import defaultdict

import sys

# ====================== 路径与环境设置 ======================

# 当前脚本目录: .../4.SAC/experiment
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 自动探测 Train / Test 目录(兼容 Test/test, Train/train) ---
def _resolve_dir(candidates):
    for rel in candidates:
        full = os.path.join(ROOT_DIR, rel)
        if os.path.isdir(full):
            return full
    # 默认返回第一个候选，后续 collect_test_images 会再做检查
    return os.path.join(ROOT_DIR, candidates[0])

TEST_ROOT = _resolve_dir(["../Test", "../test"])
TRAIN_ROOT = _resolve_dir(["../Train", "../train"])

# 构建正确的 2.distillation 路径
parent_dir = os.path.dirname(ROOT_DIR)  # .../4.SAC
distill_path = os.path.join(parent_dir, "2.distillation")
if distill_path not in sys.path:
    sys.path.insert(0, distill_path)

from clahe_env_dual_V1f import CLAHEEnvPro

DISTILL_DIR = distill_path

# 本文方法：双教师蒸馏 + QAT 学生
STUDENT_WEIGHTS = os.path.join(DISTILL_DIR, "distill_out", "student_o5_multihead.pt")
STUDENT_NORM    = os.path.join(DISTILL_DIR, "distill_out", "obs_norm_o5_multihead.json")

# SAC 教师 (单教师 baseline，可选 rich / lite)
RICH_VEC_PATH   = os.path.join(DISTILL_DIR, "weight", "vecnorm_rich.pkl")
RICH_MODEL_PATH = os.path.join(DISTILL_DIR, "weight", "sac_rich_ckpt_v1c_600000_steps.zip")
LITE_VEC_PATH   = os.path.join(DISTILL_DIR, "weight", "vecnorm_lite.pkl")
LITE_MODEL_PATH = os.path.join(DISTILL_DIR, "weight", "sac_lite_ckpt_v1f_350000_steps.zip")

# 选用哪一个 SAC 教师
SINGLE_TEACHER_TAG = "lite"  # "lite" or "rich"

DETAIL_CSV  = os.path.join(ROOT_DIR, "metrics_testset_detail.csv")
SUMMARY_CSV = os.path.join(ROOT_DIR, "metrics_testset_summary.csv")

PLOT_DIR = os.path.join(ROOT_DIR, "metrics_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

SUMMARY_STD_CSV      = os.path.join(ROOT_DIR, "metrics_testset_summary_with_std.csv")
SUMMARY_BY_SCENE_CSV = os.path.join(ROOT_DIR, "metrics_testset_summary_by_scene.csv")
TAIL_CSV             = os.path.join(ROOT_DIR, "metrics_testset_tail_stats.csv")


# 策略相关常数
MAX_STEPS       = 5
CL_MIN          = 0.1
CL_MAX          = 20.0
INIT_CL         = 5.0
DELTA_CL_CLIP   = 2.0       # 学生 ΔCL 裁剪
DELTA_CL_MAX    = 2.0       # 教师动作范围 [-2,2]
FIXED_CL_BASE   = 2.0       # 固定参数 CLAHE 的 CL=2.0 (按你的要求)


# ====================== 中文路径安全 imread ======================

def cv_imread(path: str, flags=cv2.IMREAD_GRAYSCALE):
    """
    兼容含中文路径的安全 imread:
    - 使用 np.fromfile + cv2.imdecode
    - 若文件不存在或为空, 返回 None
    """
    try:
        if not os.path.exists(path):
            return None
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception as e:
        print(f"[错误] cv_imread 失败: {path}, err={e}")
        return None


# ====================== 学生网络结构 ======================

class MLP5(nn.Module):
    def __init__(self, in_dim: int, h1: int = 128, h2: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(inplace=True),
            nn.Linear(h1, h2), nn.ReLU(inplace=True),
        )
        self.head_a   = nn.Linear(h2, 1)
        self.head_aux = nn.Linear(h2, 1)

    def forward(self, x):
        z = self.trunk(x)
        a = self.head_a(z)
        aux = self.head_aux(z)
        return a, aux


def load_student() -> Tuple[MLP5, np.ndarray, np.ndarray, float]:
    if not os.path.exists(STUDENT_WEIGHTS):
        raise FileNotFoundError(f"未找到学生权重文件: {STUDENT_WEIGHTS}")
    if not os.path.exists(STUDENT_NORM):
        raise FileNotFoundError(f"未找到学生归一化 JSON: {STUDENT_NORM}")

    import json
    with open(STUDENT_NORM, "r", encoding="utf-8") as f:
        norm_json = json.load(f)

    if "mean" not in norm_json or "std" not in norm_json:
        raise KeyError(f"{STUDENT_NORM} 中缺少 'mean' 或 'std' 字段")

    mean = np.array(norm_json["mean"], dtype=np.float32)
    std  = np.array(norm_json["std"],  dtype=np.float32)
    if mean.shape[0] != 5 or std.shape[0] != 5:
        raise ValueError(f"学生观测维度应为 5，实际 mean={mean.shape}, std={std.shape}")

    std = np.where(std < 1e-8, 1.0, std)
    delta_cl_max = float(norm_json.get("delta_cl_max", DELTA_CL_MAX))

    in_dim = mean.shape[0]
    student = MLP5(in_dim=in_dim)
    state = torch.load(STUDENT_WEIGHTS, map_location="cpu")
    student.load_state_dict(state, strict=True)
    student.to("cpu").eval()

    print(f"[信息] 双教师学生模型已加载: 输入维度 = {in_dim}, delta_cl_max = {delta_cl_max}")
    return student, mean, std, delta_cl_max


# ====================== 教师加载 & 推理 ======================

def load_single_teacher():
    if SINGLE_TEACHER_TAG not in ("lite", "rich"):
        raise ValueError("SINGLE_TEACHER_TAG 必须为 'lite' 或 'rich'")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if SINGLE_TEACHER_TAG == "lite":
        vec_path, model_path, obs_mode = LITE_VEC_PATH, LITE_MODEL_PATH, "lite"
    else:
        vec_path, model_path, obs_mode = RICH_VEC_PATH, RICH_MODEL_PATH, "rich"

    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"未找到 VecNormalize 文件: {vec_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到 SAC 教师权重: {model_path}")

    env_root = TRAIN_ROOT
    if not os.path.isdir(env_root):
        raise FileNotFoundError(f"教师环境图片根目录不存在: {env_root}")

    # 用 TRAIN_ROOT 构建一个占位环境（只用于加载归一化统计，不用于真实 roll-out）
    tmp_env = DummyVecEnv([
        lambda: CLAHEEnvPro(
            image_folder=env_root,
            max_steps=MAX_STEPS,
            obs_mode=obs_mode,
            metrics_device=device,
        )
    ])
    vec_norm = VecNormalize.load(vec_path, tmp_env)
    vec_norm.training = False
    vec_norm.norm_reward = False

    model = SAC.load(model_path, device=device)

    print(f"[信息] SAC 单教师已加载: tag={SINGLE_TEACHER_TAG}, obs_mode={obs_mode}")
    return model, vec_norm, device, obs_mode


# ====================== 指标计算工具 ======================

def calc_shannon_entropy(image: np.ndarray) -> float:
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / max(hist.sum(), 1.0)
    hist_norm = hist_norm[hist_norm > 0]
    if hist_norm.size == 0:
        return 0.0
    return float(-np.sum(hist_norm * np.log2(hist_norm)))


def calc_rms_contrast(image: np.ndarray) -> float:
    return float(image.std())


def calc_renyi_entropy(image: np.ndarray) -> float:
    hist = np.histogram(image, bins=256, range=(0, 256))[0].astype(np.float64)
    total = image.size
    if total == 0:
        return 0.0
    p = hist / total
    s = np.sum(p * p)
    if s <= 1e-12:
        return 0.0
    return float(-np.log2(s))


def calc_average_gradient(image: np.ndarray) -> float:
    """
    平均梯度:
      Gx = I(i+1,j) - I(i,j)
      Gy = I(i,j+1) - I(i,j)
      G  = sqrt((Gx^2 + Gy^2) / 2)
      G_avg = mean(G)
    """
    img = image.astype(np.float32)
    gx = img[1:, :] - img[:-1, :]
    gy = img[:, 1:] - img[:, :-1]

    gx_c = gx[:, :-1]
    gy_c = gy[:-1, :]

    grad = np.sqrt(0.5 * (gx_c ** 2 + gy_c ** 2))
    return float(np.mean(grad))


def calc_local_contrast(image: np.ndarray, win: int = 3) -> float:
    """
    平均局部对比度:
      先计算 w×w 局部均值 μ 和局部标准差 σ，
      C_ij = σ_ij / (μ_ij + eps)，再对全图平均。
    """
    img = image.astype(np.float32)
    eps = 1e-6

    mean = cv2.blur(img, (win, win))
    mean_sq = cv2.blur(img * img, (win, win))

    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)

    c_local = std / (mean + eps)
    return float(np.mean(c_local))


# ====================== 图像增强方法实现 ======================

def apply_clahe_fixed(image: np.ndarray, clip_limit: float) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    return clahe.apply(image)


def compute_rule_based_cl(image: np.ndarray) -> float:
    """
    简单经验规则: 根据亮度、方差、Rényi 熵估计一个 CL
    """
    mu = float(np.mean(image))
    var = float(np.var(image))
    H2 = float(calc_renyi_entropy(image))

    norm_mean = mu / 255.0
    norm_var  = min(var / 5000.0, 1.0)
    norm_H2   = float(np.clip((H2 - 4.0) / 4.0, 0.0, 1.0))

    score = (
        (1.0 - norm_mean) * 0.5 +
        (1.0 - norm_var)  * 0.3 +
        (1.0 - norm_H2)   * 0.2
    )

    cl = 2.0 + score * 12.0
    return float(np.clip(cl, CL_MIN, CL_MAX))


def apply_clahe_rule_based(image: np.ndarray) -> Tuple[np.ndarray, float]:
    cl = compute_rule_based_cl(image)
    enhanced = apply_clahe_fixed(image, clip_limit=cl)
    return enhanced, cl


def apply_adaptive_gamma(image: np.ndarray) -> np.ndarray:
    """
    自适应 Gamma 校正（业界常用简单增强手段之一）:
      - 根据全局亮度选择 γ < 1 (提亮) 或 γ > 1 (压暗)
    """
    img = image.astype(np.float32) / 255.0
    mu = float(img.mean())

    if mu < 0.4:
        gamma = 0.7   # 偏暗 -> 提亮
    elif mu > 0.7:
        gamma = 1.4   # 偏亮 -> 压暗
    else:
        gamma = 1.0   # 中等亮度不大改

    out = np.power(np.clip(img, 0.0, 1.0), gamma)
    out = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def apply_msr_retinex(image: np.ndarray) -> np.ndarray:
    """
    简单 MSR Retinex 灰度增强 (Single-channel):
      R = (1/K) * sum_k [ log(I) - log(G_sigma_k * I) ]
      结果归一化到 [0,255]
    """
    img = image.astype(np.float32)
    img = img + 1.0  # 避免 log(0)
    sigmas = [15, 80, 250]

    ret = np.zeros_like(img)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
        ret += (np.log(img) - np.log(blur + 1e-3))
    ret /= len(sigmas)

    # 线性拉伸到 [0,255]
    r_min, r_max = float(ret.min()), float(ret.max())
    if abs(r_max - r_min) < 1e-6:
        out = np.zeros_like(ret)
    else:
        out = (ret - r_min) / (r_max - r_min) * 255.0
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return out


def apply_clahe_with_student(
    image: np.ndarray,
    student: MLP5,
    mean: np.ndarray,
    std: np.ndarray,
    delta_cl_max: float,
    max_steps: int = MAX_STEPS,
    init_cl: float = INIT_CL,
) -> Tuple[np.ndarray, float]:
    mu = float(np.mean(image))
    var = float(np.var(image))
    H2 = float(calc_renyi_entropy(image))

    cl = float(init_cl)

    for t in range(max_steps):
        x = np.array([mu, var, H2, cl, float(t)], dtype=np.float32)
        xn = (x - mean) / std
        xt = torch.from_numpy(xn).unsqueeze(0)

        with torch.no_grad():
            y_raw, _ = student(xt)
            y_raw = float(y_raw[0, 0].item())

        delta_cl = math.tanh(y_raw) * float(delta_cl_max)
        delta_cl = float(np.clip(delta_cl, -DELTA_CL_CLIP, DELTA_CL_CLIP))

        cl = float(np.clip(cl + delta_cl, CL_MIN, CL_MAX))

    enhanced = apply_clahe_fixed(image, clip_limit=cl)
    return enhanced, cl


class IQAMetrics:
    def __init__(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"[信息] IQA 使用设备: {self.device}")
        self._brisque = pyiqa.create_metric("brisque", device=self.device)
        self._niqe   = pyiqa.create_metric("niqe",    device=self.device)

    def compute(self, gray_image: np.ndarray) -> Tuple[float, float]:
        rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        ten = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            b = float(self._brisque(ten).item())
            n = float(self._niqe(ten).item())
        return b, n


def apply_clahe_with_teacher(
    image: np.ndarray,
    model: SAC,
    vec_norm: VecNormalize,
    obs_mode: str,
    iqa: IQAMetrics,
) -> Tuple[np.ndarray, float]:
    """
    使用 SAC 教师（rich/lite）在单张图上 rollout MAX_STEPS 步，
    返回增强后图像和最终 CL。
    """
    mu = float(np.mean(image))
    var = float(np.var(image))
    H2 = float(calc_renyi_entropy(image))

    init_b, init_n = iqa.compute(image)
    cl = float(INIT_CL)
    last_b, last_n = init_b, init_n

    for t in range(MAX_STEPS):
        norm_mean    = mu / 255.0
        norm_var     = var / 10000.0
        norm_entropy = H2 / 8.0
        norm_cl      = (cl - CL_MIN) / (CL_MAX - CL_MIN)
        norm_step    = t / MAX_STEPS

        if obs_mode == "rich":
            norm_b = last_b / 100.0
            norm_n = last_n / 10.0
            obs = np.array(
                [norm_mean, norm_var, norm_entropy,
                 norm_cl, norm_step, norm_b, norm_n],
                dtype=np.float32,
            )
        else:
            obs = np.array(
                [norm_mean, norm_var, norm_entropy,
                 norm_cl, norm_step],
                dtype=np.float32,
            )

        obs = np.clip(obs, 0.0, 1.0)
        obs_norm = vec_norm.normalize_obs(obs)

        action, _ = model.predict(obs_norm, deterministic=True)
        delta_cl = float(np.clip(action[0], -DELTA_CL_MAX, DELTA_CL_MAX))

        cl = float(np.clip(cl + delta_cl, CL_MIN, CL_MAX))

        enhanced = apply_clahe_fixed(image, clip_limit=cl)
        last_b, last_n = iqa.compute(enhanced)

    return enhanced, cl


# ====================== 测试集遍历 ======================

def collect_test_images(test_root: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"测试集目录不存在: {test_root}")

    items: List[Tuple[str, str]] = []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif")

    for entry in sorted(os.listdir(test_root)):
        full = os.path.join(test_root, entry)
        if os.path.isdir(full):
            scene = entry
            for fname in sorted(os.listdir(full)):
                if fname.lower().endswith(exts):
                    items.append((scene, os.path.join(full, fname)))
        else:
            if entry.lower().endswith(exts):
                items.append(("root", full))

    if not items:
        raise ValueError(f"在 {test_root} 下没有找到任何图像文件。")

    print(f"[信息] 共收集到 {len(items)} 张测试图像。")
    return items
def summarize_array(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"n": 0}

    p10 = float(np.percentile(arr, 10))
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))

    # worst-10% mean: 对于“越小越好”的指标（BRISQUE/NIQE），worst 指最大的一端
    k = max(1, int(np.ceil(arr.size * 0.10)))
    worst10_mean = float(np.mean(np.sort(arr)[-k:]))

    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": p50,
        "p10": p10,
        "p90": p90,
        "p95": p95,
        "worst10_mean": worst10_mean,
    }

# cdf测试
# ====================== 绘图辅助函数 (修改版) ======================

def _save_cdf_plot(metric_name: str, data_by_method: Dict[str, np.ndarray], out_png: str):
    plt.figure()
    for method, arr in data_by_method.items():
        a = np.sort(arr[~np.isnan(arr)])
        if a.size == 0:
            continue
        y = np.arange(1, a.size + 1) / a.size

        # --- 修改点: 简化标签名称 ---
        # 如果是本文方法，图例只显示 "Proposed"，否则显示原名
        label_text = "Proposed" if "Proposed" in method else method

        plt.plot(a, y, label=label_text)

    plt.xlabel(metric_name)
    plt.ylabel("CDF")
    plt.grid(True, linewidth=0.5)
    plt.legend(fontsize=10)  # 稍微调大字体使其更清晰
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _save_boxplot(metric_name: str, methods: List[str], data_by_method: Dict[str, np.ndarray], out_png: str):
    plt.figure()
    data = [data_by_method[m][~np.isnan(data_by_method[m])] for m in methods]

    # --- 修改点: 简化标签名称 ---
    # 生成用于 x 轴显示的短标签列表
    clean_labels = ["Proposed" if "Proposed" in m else m for m in methods]

    plt.boxplot(data, tick_labels=clean_labels, showfliers=False)
    plt.ylabel(metric_name)

    # 因为名字变短了，旋转角度可以减小或去掉，这里保留一点角度以防万一
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ====================== 主流程 ======================

def main():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示
    print("========== 在测试集上评估 8 种方法的客观指标 ==========")
    print(f"[信息] 当前脚本目录: {ROOT_DIR}")
    print(f"[信息] 测试集根目录: {TEST_ROOT}")

    img_items = collect_test_images(TEST_ROOT)

    iqa = IQAMetrics()

    # 学生 & 教师
    student, stu_mean, stu_std, stu_delta = load_student()
    teacher_model, teacher_vec, teacher_device, teacher_obs_mode = load_single_teacher()

    methods = [
        "Input",
        "Global HE",
        "Fixed CLAHE (CL=2.0)",
        "Rule-based Adaptive CLAHE",
        "Adaptive Gamma Correction",
        "MSR Retinex",
        "SAC Single-Teacher",
        "Proposed: Dual-Teacher Distill + QAT Student",
    ]

    stats: Dict[str, Dict[str, List[float]]] = {
        m: {
            "brisque": [],
            "niqe": [],
            "entropy": [],
            "rms": [],
            "grad": [],
            "local_contrast": [],
        }
        for m in methods
    }

    detail_rows: List[Dict[str, object]] = []

    for idx, (scene, img_path) in enumerate(img_items, 1):
        img = cv_imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            if not os.path.exists(img_path):
                print(f"[警告] 文件不存在，跳过: {img_path}")
            else:
                print(f"[警告] OpenCV 无法解码图像，跳过: {img_path}")
            continue

        img_name = os.path.basename(img_path)
        print(f"[{idx}/{len(img_items)}] 处理 {scene}/{img_name} ...")

        variants: Dict[str, Tuple[np.ndarray, Dict[str, float]]] = {}

        variants["Input"] = (img, {"CL": 0.0})

        he_img = cv2.equalizeHist(img)
        variants["Global HE"] = (he_img, {"CL": 0.0})

        clahe_fixed_img = apply_clahe_fixed(img, clip_limit=FIXED_CL_BASE)
        variants["Fixed CLAHE (CL=2.0)"] = (clahe_fixed_img, {"CL": FIXED_CL_BASE})

        rule_img, rule_cl = apply_clahe_rule_based(img)
        variants["Rule-based Adaptive CLAHE"] = (rule_img, {"CL": rule_cl})

        gamma_img = apply_adaptive_gamma(img)
        variants["Adaptive Gamma Correction"] = (gamma_img, {"CL": 0.0})

        msr_img = apply_msr_retinex(img)
        variants["MSR Retinex"] = (msr_img, {"CL": 0.0})

        teacher_img, teacher_cl = apply_clahe_with_teacher(
            img, teacher_model, teacher_vec, teacher_obs_mode, iqa
        )
        variants["SAC Single-Teacher"] = (teacher_img, {"CL": teacher_cl})

        dual_img, dual_cl = apply_clahe_with_student(
            img, student, stu_mean, stu_std, stu_delta,
            max_steps=MAX_STEPS, init_cl=INIT_CL
        )
        variants["Proposed: Dual-Teacher Distill + QAT Student"] = (dual_img, {"CL": dual_cl})

        for method, (out_img, extra) in variants.items():
            # IQA
            b, n = iqa.compute(out_img)
            # 全局信息熵
            h = calc_shannon_entropy(out_img)
            # 全局 RMS 对比度
            c_rms = calc_rms_contrast(out_img)
            # 平均梯度
            g_avg = calc_average_gradient(out_img)
            # 局部对比度
            c_local = calc_local_contrast(out_img, win=3)

            stats[method]["brisque"].append(b)
            stats[method]["niqe"].append(n)
            stats[method]["entropy"].append(h)
            stats[method]["rms"].append(c_rms)
            stats[method]["grad"].append(g_avg)
            stats[method]["local_contrast"].append(c_local)

            row = {
                "scene": scene,
                "image": img_name,
                "method": method,
                "CL": float(extra.get("CL", 0.0)),
                "Q_BRISQUE": b,
                "Q_NIQE": n,
                "H_Shannon": h,
                "C_RMS": c_rms,
                "G_avg": g_avg,
                "C_local": c_local,
            }
            detail_rows.append(row)

    if not detail_rows:
        print("[错误] 没有任何有效结果，请检查测试集路径和图像文件。")
        return

    # 写详细 CSV
    fieldnames = [
        "scene", "image", "method", "CL",
        "Q_BRISQUE", "Q_NIQE",
        "H_Shannon", "C_RMS",
        "G_avg", "C_local",
    ]
    with open(DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"[信息] 详细结果已写入: {DETAIL_CSV}")

    # ====================== 基于 detail_rows 的增强统计（M1 修复核心） ======================

    # 1) 按 method 汇总数组（更方便算 std/分位数/CDF）
    metrics = ["Q_BRISQUE", "Q_NIQE", "H_Shannon", "C_RMS", "G_avg", "C_local"]
    by_method = {m: {k: [] for k in metrics} for m in methods}
    by_scene_method = defaultdict(lambda: {k: [] for k in metrics})  # key=(scene,method)

    for r in detail_rows:
        m = r["method"]
        sc = r["scene"]
        for k in metrics:
            by_method[m][k].append(float(r[k]))
            by_scene_method[(sc, m)][k].append(float(r[k]))

    # 2) 全测试集：mean±std + 分位数 + worst10
    summary_std_rows = []
    tail_rows = []

    for m in methods:
        row = {"method": m}
        for k in metrics:
            s = summarize_array(np.array(by_method[m][k], dtype=np.float64))
            row[f"{k}_mean"] = s.get("mean", np.nan)
            row[f"{k}_std"]  = s.get("std",  np.nan)
        summary_std_rows.append(row)

        # 重点输出 BRISQUE/NIQE 的尾部统计，便于检查无参考质量指标的尾部风险。
        for k in ["Q_BRISQUE", "Q_NIQE"]:
            s = summarize_array(np.array(by_method[m][k], dtype=np.float64))
            tail_rows.append({
                "method": m,
                "metric": k,
                "n": s.get("n", 0),
                "mean": s.get("mean", np.nan),
                "std": s.get("std", np.nan),
                "median": s.get("median", np.nan),
                "p10": s.get("p10", np.nan),
                "p90": s.get("p90", np.nan),
                "p95": s.get("p95", np.nan),
                "worst10_mean": s.get("worst10_mean", np.nan),
            })

    with open(SUMMARY_STD_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_std_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_std_rows)
    print("[信息] 均值±std 汇总已写入:")
    print(f"  {os.path.abspath(SUMMARY_STD_CSV)}")

    with open(TAIL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method","metric","n","mean","std","median","p10","p90","p95","worst10_mean"],
        )
        writer.writeheader()
        writer.writerows(tail_rows)
    print(f"[信息] 尾部/分位数统计已写入: {TAIL_CSV}")

    # 3) 分场景统计：每个 scene × method 输出 BRISQUE/NIQE mean±std
    by_scene_rows = []
    scenes = sorted({r["scene"] for r in detail_rows})

    for sc in scenes:
        for m in methods:
            key = (sc, m)
            if key not in by_scene_method:
                continue
            b = summarize_array(np.array(by_scene_method[key]["Q_BRISQUE"], dtype=np.float64))
            n = summarize_array(np.array(by_scene_method[key]["Q_NIQE"], dtype=np.float64))
            by_scene_rows.append({
                "scene": sc,
                "method": m,
                "n": b.get("n", 0),
                "Q_BRISQUE_mean": b.get("mean", np.nan),
                "Q_BRISQUE_std":  b.get("std", np.nan),
                "Q_NIQE_mean":    n.get("mean", np.nan),
                "Q_NIQE_std":     n.get("std", np.nan),
            })

    with open(SUMMARY_BY_SCENE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene","method","n","Q_BRISQUE_mean","Q_BRISQUE_std","Q_NIQE_mean","Q_NIQE_std"],
        )
        writer.writeheader()
        writer.writerows(by_scene_rows)
    print(f"[信息] 分场景汇总已写入: {SUMMARY_BY_SCENE_CSV}")

    # 4) 分布图：CDF + Boxplot（BRISQUE/NIQE）
    brisque_by_method = {m: np.array(by_method[m]["Q_BRISQUE"], dtype=np.float64) for m in methods}
    niqe_by_method    = {m: np.array(by_method[m]["Q_NIQE"], dtype=np.float64)    for m in methods}

    _save_cdf_plot("BRISQUE (lower is better)", brisque_by_method,
                   os.path.join(PLOT_DIR, "cdf_brisque.png"))
    _save_cdf_plot("NIQE (lower is better)", niqe_by_method,
                   os.path.join(PLOT_DIR, "cdf_niqe.png"))

    _save_boxplot("BRISQUE (lower is better)", methods, brisque_by_method,
                  os.path.join(PLOT_DIR, "box_brisque.png"))
    _save_boxplot("NIQE (lower is better)", methods, niqe_by_method,
                  os.path.join(PLOT_DIR, "box_niqe.png"))

    print(f"[信息] 分布图已输出到目录: {PLOT_DIR}")



    def mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    summary_rows: List[Dict[str, object]] = []
    print("\n===== 各方法在整个测试集上的平均指标 =====")
    print("(可直接对应 LaTeX 表中的一行)")
    print("Method\tQ_BRISQUE↓\tQ_NIQE↓\tH_Shannon↑\tC_RMS↑\tG_avg↑\tC_local↑")


    for method in methods:
        m_b  = mean(stats[method]["brisque"])
        m_n  = mean(stats[method]["niqe"])
        m_h  = mean(stats[method]["entropy"])
        m_c  = mean(stats[method]["rms"])
        m_g  = mean(stats[method]["grad"])
        m_cl = mean(stats[method]["local_contrast"])

        summary_rows.append({
            "method": method,
            "Q_BRISQUE": m_b,
            "Q_NIQE": m_n,
            "H_Shannon": m_h,
            "C_RMS": m_c,
            "G_avg": m_g,
            "C_local": m_cl,
        })

        print(f"{method}\t{m_b:.2f}\t{m_n:.2f}\t{m_h:.3f}\t{m_c:.3f}\t{m_g:.3f}\t{m_cl:.3f}")

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "Q_BRISQUE", "Q_NIQE",
                "H_Shannon", "C_RMS",
                "G_avg", "C_local",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n[信息] 汇总结果已写入: {SUMMARY_CSV}")
    print("========== 评估完成 ==========")


if __name__ == "__main__":
    main()
