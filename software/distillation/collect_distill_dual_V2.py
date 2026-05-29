# 文件名: collect_distill_dual_V2.py
"""
双教师蒸馏数据采样 (Rich + Lite) - 带消融模式

改动要点:
- MODE = "full" / "wo_rich" / "wo_traj"
  * full     : Rich + Lite 都 rollout, 按最终 BRISQUE 选更优老师的整条轨迹写入 CSV
  * wo_rich  : 仅使用 Lite 教师, 所有轨迹均由 Lite 生成
  * wo_traj  : Rich + Lite 都 rollout, 不做后验轨迹优选, 两个老师的所有轨迹等权合并写入 CSV
- 本脚本只负责采样蒸馏数据, 输出到 distill_out/*.csv
- 字段兼容 train_student_distill_o5_multihead_V2.py 需要的列:
    mu, var, H2, CL_t, step_t, a_teacher

使用方式(建议在 4.SAC/2.distillation 目录下运行):
    python collect_distill_dual_V2.py
    (根据需要修改下方 MODE / EVAL_IMAGE_FOLDER 等配置)
"""

import os
import csv

import cv2
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from clahe_env_dual_V1f import calculate_renyi_entropy, apply_clahe, CLAHEEnvPro

# ----------------- 配置 -----------------
# 蒸馏模式: "full" / "wo_rich" / "wo_traj"
MODE = "wo_traj"

# 用哪批图来采样蒸馏数据
EVAL_IMAGE_FOLDER = "../Train"          # 你原来的训练图片根目录

OUT_DIR    = "distill_out"
if MODE == "full":
    OUT_CSV = os.path.join(OUT_DIR, "distill_data.csv")
elif MODE == "wo_rich":
    OUT_CSV = os.path.join(OUT_DIR, "distill_data_wo_rich.csv")
elif MODE == "wo_traj":
    OUT_CSV = os.path.join(OUT_DIR, "distill_data_wo_traj.csv")
else:
    raise ValueError(f"未知 MODE: {MODE}")

RICH_VEC_PATH   = "weight/vecnorm_rich.pkl"
RICH_MODEL_PATH = "weight/sac_rich_ckpt_v1c_600000_steps.zip"

LITE_VEC_PATH   = "weight/vecnorm_lite.pkl"
LITE_MODEL_PATH = "weight/sac_lite_ckpt_v1f_350000_steps.zip"

MAX_STEPS    = 5
CL_MIN       = 0.1
CL_MAX       = 20.0
DELTA_CL_MAX = 2.0   # SAC 动作边界

INIT_CL      = 5.0   # 所有 episode 的初始 CL
IQA_MAX_SIZE = 384   # IQA 输入最长边（越小越快）

os.makedirs(OUT_DIR, exist_ok=True)
torch.set_num_threads(1)


# ----------------- 工具函数 -----------------
def list_images(folder):
    exts = (".jpg", ".png", ".jpeg", ".bmp", ".tif")
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]


def calc_features(img_gray):
    mu  = float(np.mean(img_gray))
    var = float(np.var(img_gray))
    H2  = float(calculate_renyi_entropy(img_gray))
    return mu, var, H2


# ---------- CPU 上的 IQA（带下采样） ----------
def create_iqa_metrics(device="cpu"):
    import pyiqa
    brisque_metric = pyiqa.create_metric("brisque", device=device)
    niqe_metric    = pyiqa.create_metric("niqe",    device=device)
    return brisque_metric, niqe_metric


def iqa_cpu_downsample(img_gray, brisque_metric, niqe_metric, max_size=384):
    """在CPU上计算BRISQUE/NIQE, 先把图像最长边缩到<=max_size, 加速计算"""
    h, w = img_gray.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img_small = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_small = img_gray

    rgb = cv2.cvtColor(img_small, cv2.COLOR_GRAY2RGB)
    ten = torch.tensor(rgb / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        b = float(brisque_metric(ten).item())
        n = float(niqe_metric(ten).item())
    return b, n


# ---------- 单个教师 rollout ----------
def rollout_one_teacher(
    img_gray,
    mu, var, H2,
    brisque_metric,
    niqe_metric,
    vec_norm: VecNormalize,
    model: SAC,
    teacher_tag: str,
):
    """对于给定图像和一个教师(Rich/Lite), 从 INIT_CL 出发 rollout MAX_STEPS 步。"""
    # 初始 IQA
    init_b, init_n = iqa_cpu_downsample(img_gray, brisque_metric, niqe_metric, max_size=IQA_MAX_SIZE)
    last_b, last_n = init_b, init_n

    cl = INIT_CL
    records = []

    for t in range(MAX_STEPS):
        # 构造观测
        norm_mean    = mu / 255.0
        norm_var     = var / 10000.0
        norm_entropy = H2 / 8.0
        norm_cl      = (cl - CL_MIN) / (CL_MAX - CL_MIN)
        norm_step    = t / MAX_STEPS

        if teacher_tag == "rich":
            # rich: 7维 [mu,var,H2,CL,step,BRISQUE,NIQE]
            norm_b = last_b / 100.0
            norm_n = last_n / 10.0
            obs = np.array(
                [norm_mean, norm_var, norm_entropy,
                 norm_cl, norm_step, norm_b, norm_n],
                dtype=np.float32,
            )
        else:
            # lite: 5维 [mu,var,H2,CL,step]
            obs = np.array(
                [norm_mean, norm_var, norm_entropy,
                 norm_cl, norm_step],
                dtype=np.float32,
            )

        obs = np.clip(obs, 0.0, 1.0)
        obs_norm = vec_norm.normalize_obs(obs)

        # SAC 决策
        action, _ = model.predict(obs_norm, deterministic=True)
        delta_cl = float(np.clip(action[0], -DELTA_CL_MAX, DELTA_CL_MAX))

        cl_t = cl  # 当前步CL
        cl = float(np.clip(cl + delta_cl, CL_MIN, CL_MAX))

        # CLAHE + IQA（CPU+降采样）
        enh = apply_clahe(img_gray, cl)
        new_b, new_n = iqa_cpu_downsample(enh, brisque_metric, niqe_metric, max_size=IQA_MAX_SIZE)

        rec = dict(
            img="",          # 外层填
            teacher=teacher_tag,
            mu=mu,
            var=var,
            H2=H2,
            CL_t=cl_t,
            step_t=t,
            BRISQUE=new_b,
            NIQE=new_n,
            a_teacher=delta_cl,
            CL_next=cl,
            init_b=init_b,
            init_n=init_n,
        )
        records.append(rec)

        last_b, last_n = new_b, new_n

    final_brisque = records[-1]["BRISQUE"]
    return records, final_brisque


# ---------- 主流程 ----------
def main():
    print(f"==> 双教师蒸馏数据采样启动, MODE = {MODE}")
    print(f"==> 图像根目录: {EVAL_IMAGE_FOLDER}")
    print(f"==> 输出 CSV: {OUT_CSV}")

    # ---- 检查文件 ----
    if MODE in ("full", "wo_traj"):
        if (not os.path.exists(RICH_VEC_PATH)) or (not os.path.exists(RICH_MODEL_PATH)):
            raise FileNotFoundError("Rich 模型或 vecnorm 文件缺失。")

    if not os.path.exists(LITE_VEC_PATH) or not os.path.exists(LITE_MODEL_PATH):
        raise FileNotFoundError("Lite 模型或 vecnorm 文件缺失。")

    if not os.path.isdir(EVAL_IMAGE_FOLDER):
        raise FileNotFoundError(f"评估图片文件夹不存在: {EVAL_IMAGE_FOLDER}")

    img_paths = list_images(EVAL_IMAGE_FOLDER)
    if not img_paths:
        raise ValueError(f"文件夹 {EVAL_IMAGE_FOLDER} 中没有图片。")

    print(f"共发现 {len(img_paths)} 张图像用于双教师采样。")

    device = "cpu"   # 当前环境只有CPU
    brisque_metric, niqe_metric = create_iqa_metrics(device=device)

    # ---- 加载 VecNormalize + SAC 模型 ----
    vec_rich = None
    model_rich = None

    if MODE in ("full", "wo_traj"):
        print("加载 Rich 模型和 VecNormalize (CPU)...")
        tmp_env_rich = DummyVecEnv([lambda: CLAHEEnvPro(
            image_folder=EVAL_IMAGE_FOLDER,
            max_steps=MAX_STEPS,
            obs_mode="rich",
            metrics_device=device,
        )])
        vec_rich = VecNormalize.load(RICH_VEC_PATH, tmp_env_rich)
        vec_rich.training = False
        vec_rich.norm_reward = False
        model_rich = SAC.load(RICH_MODEL_PATH, device=device)

    print("加载 Lite 模型和 VecNormalize (CPU)...")
    tmp_env_lite = DummyVecEnv([lambda: CLAHEEnvPro(
        image_folder=EVAL_IMAGE_FOLDER,
        max_steps=MAX_STEPS,
        obs_mode="lite",
        metrics_device=device,
    )])
    vec_lite = VecNormalize.load(LITE_VEC_PATH, tmp_env_lite)
    vec_lite.training = False
    vec_lite.norm_reward = False
    model_lite = SAC.load(LITE_MODEL_PATH, device=device)

    all_records = []

    # ---- 遍历图片 ----
    for idx, path in enumerate(sorted(img_paths), 1):
        name = os.path.basename(path)
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"[警告] 读取失败，跳过: {path}")
            continue

        mu, var, H2 = calc_features(img_gray)

        if MODE == "wo_rich":
            # 仅 Lite 教师
            rec_lite, fb_lite = rollout_one_teacher(
                img_gray, mu, var, H2,
                brisque_metric, niqe_metric,
                vec_lite, model_lite,
                teacher_tag="lite",
            )
            for r in rec_lite:
                r["img"] = name
                all_records.append(r)

            print(f"[{idx}/{len(img_paths)}] {name}: 仅使用 Lite, 最终 BRISQUE_lite={fb_lite:.2f}")
        else:
            # Rich 与 Lite 都 rollout
            rec_rich, fb_rich = rollout_one_teacher(
                img_gray, mu, var, H2,
                brisque_metric, niqe_metric,
                vec_rich, model_rich,
                teacher_tag="rich",
            )
            rec_lite, fb_lite = rollout_one_teacher(
                img_gray, mu, var, H2,
                brisque_metric, niqe_metric,
                vec_lite, model_lite,
                teacher_tag="lite",
            )

            if MODE == "full":
                # 按最终 BRISQUE 选老师
                if fb_rich <= fb_lite:
                    chosen = rec_rich
                    chosen_tag = "rich"
                else:
                    chosen = rec_lite
                    chosen_tag = "lite"

                for r in chosen:
                    r["img"] = name
                    all_records.append(r)

                print(
                    f"[{idx}/{len(img_paths)}] {name}: "
                    f"BRISQUE_rich={fb_rich:.2f}, BRISQUE_lite={fb_lite:.2f} -> 采用 {chosen_tag}"
                )
            elif MODE == "wo_traj":
                # 不做轨迹优选, 两个老师的轨迹等权合并
                for r in rec_rich:
                    r["img"] = name
                    all_records.append(r)
                for r in rec_lite:
                    r["img"] = name
                    all_records.append(r)

                print(
                    f"[{idx}/{len(img_paths)}] {name}: "
                    f"BRISQUE_rich={fb_rich:.2f}, BRISQUE_lite={fb_lite:.2f} -> 写入 rich+lite 两条轨迹"
                )

    # ---- 写 CSV ----
    if not all_records:
        print("没有有效样本, 不写出 CSV。")
        return

    fieldnames = [
        "img", "teacher",
        "mu", "var", "H2",
        "CL_t", "step_t",
        "BRISQUE", "NIQE",
        "a_teacher", "CL_next",
        "init_b", "init_n"
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n完成！共写入 {len(all_records)} 条样本到 {OUT_CSV}")
    print("后续可以直接用 train_student_distill_o5_multihead_V2.py 继续学生训练。")


if __name__ == "__main__":
    main()
