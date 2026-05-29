# 文件名: evaluate_teachers_and_student_V2.py
"""
评估 Rich-Teacher / Lite-Teacher / 学生模型 的效果

- 对同一批图像：
  * Rich 教师：7 维观测 + vecnorm_rich + sac_rich_ckpt...
  * Lite 教师：5 维观测 + vecnorm_lite + sac_lite_ckpt...
  * 学生：5 维输入 [mu,var,H2,CL_t,step_t] + obs_norm_o5_multihead.json + student_o5_multihead.pt

- 统一从 INIT_CL 出发，rollout MAX_STEPS 步
- 统计：
  * 每个模型的平均 BRISQUE/NIQE/清晰度/对比度
  * 学生 vs “更优老师” 的 CL 和 BRISQUE 偏差
"""

import os
import json
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from clahe_env_dual_V1f import CLAHEEnvPro, calculate_renyi_entropy, apply_clahe

# ------------------ 配置区 ------------------
EVAL_IMAGE_FOLDER = "../Train"   # 用哪一批图来评估（可以改成 "evaluate" 等）

# Rich 教师
RICH_VEC_PATH   = "weight/vecnorm_rich.pkl"
RICH_MODEL_PATH = "weight/sac_rich_ckpt_v1c_600000_steps.zip"

# Lite 教师
LITE_VEC_PATH   = "weight/vecnorm_lite.pkl"
LITE_MODEL_PATH = "weight/sac_lite_ckpt_v1f_350000_steps.zip"

# 学生（多头版 o5）
STUDENT_WEIGHTS = os.path.join("distill_out", "student_o5_multihead.pt")
# ✅ 这里应该指向 obs_norm，而不是 train_report
STUDENT_NORM    = os.path.join("distill_out", "obs_norm_o5_multihead.json")
# 如果你想顺带用到报告，也可以单独记一下：
STUDENT_REPORT  = os.path.join("distill_out", "train_report_o5_multihead_V2.json")

# 统一控制参数（要和训练/蒸馏保持一致）
MAX_STEPS    = 5
CL_MIN       = 0.1
CL_MAX       = 20.0
DELTA_CL_MAX = 2.0
INIT_CL      = 5.0       # 所有模型统一初始 CL

OUT_CSV      = os.path.join("distill_out", "eval_teachers_student_V2.csv")

torch.set_num_threads(1)


# ------------------ 工具函数 ------------------
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


# ---- 学生网络结构（与 train_student_distill_o5_multihead_V2.py 保持一致）----
class MLP5(nn.Module):
    """
    多头学生：trunk 提取特征，head_a 输出 ΔCL，对应 a_teacher
    """
    def __init__(self, in_dim, h1=128, h2=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(inplace=True),
            nn.Linear(h1, h2), nn.ReLU(inplace=True),
        )
        self.head_a = nn.Linear(h2, 1)  # ΔCL head
        # 可能还有其它 head（比如预测 BRISQUE），这里不影响加载 state_dict
        self.head_aux = nn.Linear(h2, 1)

    def forward(self, x):
        z = self.trunk(x)
        a = self.head_a(z)
        aux = self.head_aux(z)
        return a, aux


def rollout_teacher(
    img_gray,
    mu, var, H2,
    iqa_env: CLAHEEnvPro,
    vec_norm: VecNormalize,
    model: SAC,
    tag: str,
):
    """
    在单张图上，用某一个教师策略 rollout MAX_STEPS 步。
    返回:
      stats: dict (包含 final_iqa 等)
      traj : 每一步的记录列表（可用于分析/可视化）
    """
    init_b, init_n = iqa_env._iqa(img_gray)
    init_s = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    init_c = float(img_gray.std())

    cl = float(INIT_CL)
    last_b, last_n = init_b, init_n

    traj = []

    for t in range(MAX_STEPS):
        # 构造观测
        norm_mean    = mu / 255.0
        norm_var     = var / 10000.0
        norm_entropy = H2 / 8.0
        norm_cl      = (cl - CL_MIN) / (CL_MAX - CL_MIN)
        norm_step    = t / MAX_STEPS

        if tag == "rich":
            # 7 维
            norm_b = last_b / 100.0
            norm_n = last_n / 10.0
            obs = np.array(
                [norm_mean, norm_var, norm_entropy,
                 norm_cl, norm_step, norm_b, norm_n],
                dtype=np.float32,
            )
        else:
            # lite: 5 维
            obs = np.array(
                [norm_mean, norm_var, norm_entropy,
                 norm_cl, norm_step],
                dtype=np.float32,
            )

        obs = np.clip(obs, 0.0, 1.0)
        obs_norm = vec_norm.normalize_obs(obs)

        # SAC 策略
        action, _ = model.predict(obs_norm, deterministic=True)
        delta_cl = float(np.clip(action[0], -DELTA_CL_MAX, DELTA_CL_MAX))

        cl_t = cl
        cl = float(np.clip(cl + delta_cl, CL_MIN, CL_MAX))

        # CLAHE + IQA
        enh = apply_clahe(img_gray, cl)
        new_b, new_n = iqa_env._iqa(enh)
        new_s = float(cv2.Laplacian(enh, cv2.CV_64F).var())
        new_c = float(enh.std())

        traj.append(dict(
            CL_t=cl_t,
            step_t=t,
            CL_next=cl,
            delta_cl=delta_cl,
            BRISQUE=new_b,
            NIQE=new_n,
            sharp=new_s,
            contrast=new_c,
        ))

        last_b, last_n = new_b, new_n

    final = traj[-1]
    stats = dict(
        init_b=init_b, init_n=init_n,
        init_s=init_s, init_c=init_c,
        final_b=final["BRISQUE"],
        final_n=final["NIQE"],
        final_s=final["sharp"],
        final_c=final["contrast"],
        final_cl=final["CL_next"],
    )
    return stats, traj


def rollout_student(
    img_gray,
    mu, var, H2,
    iqa_env: CLAHEEnvPro,
    student: nn.Module,
    norm_json: dict,
):
    """
    学生只看 5 维 [mu,var,H2,CL_t,step_t] （未归一化），再用 obs_norm_o5_multihead.json 中的 mean/std 做标准化。
    """
    if "mean" not in norm_json or "std" not in norm_json:
        raise KeyError(
            "obs_norm_o5_multihead.json 中未找到 'mean' 或 'std' 字段。\n"
            "很可能你把 STUDENT_NORM 指向了 train_report_o5_multihead_V2.json，"
            "请确认配置区 STUDENT_NORM 使用的是 obs_norm_o5_multihead.json。"
        )

    mean = np.array(norm_json["mean"], dtype=np.float32)
    std  = np.array(norm_json["std"],  dtype=np.float32)
    delta_cl_max = float(norm_json.get("delta_cl_max", DELTA_CL_MAX))

    init_b, init_n = iqa_env._iqa(img_gray)
    init_s = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    init_c = float(img_gray.std())

    cl = float(INIT_CL)
    last_b, last_n = init_b, init_n

    traj = []

    for t in range(MAX_STEPS):
        x = np.array([mu, var, H2, cl, float(t)], dtype=np.float32)
        xn = (x - mean) / std
        xt = torch.from_numpy(xn).unsqueeze(0)  # [1,5]

        with torch.no_grad():
            y_raw, _ = student(xt)
            y_raw = y_raw[0, 0].item()
        delta_cl = float(np.tanh(y_raw) * delta_cl_max)
        delta_cl = float(np.clip(delta_cl, -DELTA_CL_MAX, DELTA_CL_MAX))

        cl_t = cl
        cl = float(np.clip(cl + delta_cl, CL_MIN, CL_MAX))

        enh = apply_clahe(img_gray, cl)
        new_b, new_n = iqa_env._iqa(enh)
        new_s = float(cv2.Laplacian(enh, cv2.CV_64F).var())
        new_c = float(enh.std())

        traj.append(dict(
            CL_t=cl_t,
            step_t=t,
            CL_next=cl,
            delta_cl=delta_cl,
            BRISQUE=new_b,
            NIQE=new_n,
            sharp=new_s,
            contrast=new_c,
        ))

        last_b, last_n = new_b, new_n

    final = traj[-1]
    stats = dict(
        init_b=init_b, init_n=init_n,
        init_s=init_s, init_c=init_c,
        final_b=final["BRISQUE"],
        final_n=final["NIQE"],
        final_s=final["sharp"],
        final_c=final["contrast"],
        final_cl=final["CL_next"],
    )
    return stats, traj


def main():
    # ---------- 检查文件 ----------
    if not os.path.isdir(EVAL_IMAGE_FOLDER):
        raise FileNotFoundError(f"评估图片目录不存在: {EVAL_IMAGE_FOLDER}")
    if not os.path.exists(RICH_VEC_PATH) or not os.path.exists(RICH_MODEL_PATH):
        raise FileNotFoundError("缺少 Rich 教师的 vecnorm 或模型文件。")
    if not os.path.exists(LITE_VEC_PATH) or not os.path.exists(LITE_MODEL_PATH):
        raise FileNotFoundError("缺少 Lite 教师的 vecnorm 或模型文件。")
    if not os.path.exists(STUDENT_WEIGHTS) or not os.path.exists(STUDENT_NORM):
        raise FileNotFoundError("缺少学生模型权重或归一化 json。")

    img_paths = list_images(EVAL_IMAGE_FOLDER)
    if not img_paths:
        raise ValueError(f"目录 {EVAL_IMAGE_FOLDER} 中没有图片。")
    print(f"共 {len(img_paths)} 张图用于评估。")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- IQA helper 环境 ----------
    iqa_env = CLAHEEnvPro(
        image_folder=EVAL_IMAGE_FOLDER,
        max_steps=MAX_STEPS,
        obs_mode="rich",           # 用 rich 方便 _iqa
        metrics_device=device,
    )

    # ---------- 加载 Rich 教师 ----------
    print("加载 Rich 教师...")
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

    # ---------- 加载 Lite 教师 ----------
    print("加载 Lite 教师...")
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

    # ---------- 加载学生 ----------
    print("加载学生模型...")
    with open(STUDENT_NORM, "r", encoding="utf-8") as f:
        norm_json = json.load(f)
    if "mean" not in norm_json:
        raise KeyError(
            f"{STUDENT_NORM} 中未找到 'mean' 字段。\n"
            "请确认它是 obs_norm_o5_multihead.json 而不是 train_report_o5_multihead_V2.json。"
        )
    in_dim = len(norm_json["mean"])
    student = MLP5(in_dim=in_dim)
    sd = torch.load(STUDENT_WEIGHTS, map_location="cpu")
    student.load_state_dict(sd, strict=True)
    student.to("cpu").eval()   # 学生参数量小，放 CPU 即可

    # ---------- 逐图评估 ----------
    records = []
    rich_bs, lite_bs, stu_bs = [], [], []
    rich_ns, lite_ns, stu_ns = [], [], []
    rich_ss, lite_ss, stu_ss = [], [], []
    rich_cs, lite_cs, stu_cs = [], [], []

    cl_diff_list = []
    b_diff_list  = []

    for idx, p in enumerate(img_paths, 1):
        name = os.path.basename(p)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[跳过] 读取失败: {p}")
            continue

        mu, var, H2 = calc_features(img)

        # 三个模型分别 rollout
        stats_rich, _ = rollout_teacher(
            img, mu, var, H2, iqa_env, vec_rich, model_rich, tag="rich"
        )
        stats_lite, _ = rollout_teacher(
            img, mu, var, H2, iqa_env, vec_lite, model_lite, tag="lite"
        )
        stats_stu, _  = rollout_student(
            img, mu, var, H2, iqa_env, student, norm_json
        )

        # 汇总均值用
        rich_bs.append(stats_rich["final_b"]); lite_bs.append(stats_lite["final_b"]); stu_bs.append(stats_stu["final_b"])
        rich_ns.append(stats_rich["final_n"]); lite_ns.append(stats_lite["final_n"]); stu_ns.append(stats_stu["final_n"])
        rich_ss.append(stats_rich["final_s"]); lite_ss.append(stats_lite["final_s"]); stu_ss.append(stats_stu["final_s"])
        rich_cs.append(stats_rich["final_c"]); lite_cs.append(stats_lite["final_c"]); stu_cs.append(stats_stu["final_c"])

        # 选更优老师（最终 BRISQUE 更低）
        if stats_rich["final_b"] <= stats_lite["final_b"]:
            best = stats_rich
            best_tag = "rich"
        else:
            best = stats_lite
            best_tag = "lite"

        cl_diff = abs(stats_stu["final_cl"] - best["final_cl"])
        b_diff  = stats_stu["final_b"] - best["final_b"]

        cl_diff_list.append(cl_diff)
        b_diff_list.append(b_diff)

        records.append(dict(
            img=name,
            teacher_best=best_tag,
            rich_final_cl=stats_rich["final_cl"],
            lite_final_cl=stats_lite["final_cl"],
            student_final_cl=stats_stu["final_cl"],
            rich_final_brisque=stats_rich["final_b"],
            lite_final_brisque=stats_lite["final_b"],
            student_final_brisque=stats_stu["final_b"],
            rich_final_niqe=stats_rich["final_n"],
            lite_final_niqe=stats_lite["final_n"],
            student_final_niqe=stats_stu["final_n"],
            cl_diff_vs_best=cl_diff,
            brisque_diff_vs_best=b_diff,
        ))

        print(f"[{idx}/{len(img_paths)}] {name}: "
              f"B_rich={stats_rich['final_b']:.2f}, "
              f"B_lite={stats_lite['final_b']:.2f}, "
              f"B_stu={stats_stu['final_b']:.2f}, "
              f"best={best_tag}, |ΔCL|={cl_diff:.3f}, ΔB={b_diff:.2f}")

    if not records:
        print("没有有效评估结果，退出。")
        return

    # ---------- 全局统计 ----------
    def mean(x): return float(np.mean(x)) if len(x) > 0 else float("nan")

    print("\n===== 全局均值指标 =====")
    print(f"Rich   : BRISQUE={mean(rich_bs):.2f}, NIQE={mean(rich_ns):.2f}, sharp={mean(rich_ss):.1f}, contrast={mean(rich_cs):.2f}")
    print(f"Lite   : BRISQUE={mean(lite_bs):.2f}, NIQE={mean(lite_ns):.2f}, sharp={mean(lite_ss):.1f}, contrast={mean(lite_cs):.2f}")
    print(f"Student: BRISQUE={mean(stu_bs):.2f}, NIQE={mean(stu_ns):.2f}, sharp={mean(stu_ss):.1f}, contrast={mean(stu_cs):.2f}")

    print("\n===== 学生 vs 最优老师 =====")
    print(f"平均 |CL_final(Student) - CL_final(Teacher_best)| = {mean(cl_diff_list):.3f}")
    print(f"平均 (BRISQUE_Student - BRISQUE_Teacher_best)    = {mean(b_diff_list):.2f}")

    # ---------- 写出 CSV ----------
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"\n详细结果已保存到: {OUT_CSV}")


if __name__ == "__main__":
    main()
