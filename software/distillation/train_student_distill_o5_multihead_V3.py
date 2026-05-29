# 文件名: train_student_distill_o5_multihead_V2.py
# 用法：
#   1) 手动修改下方 MODE 变量为 "full" / "wo_rich" / "wo_traj" / "wo_qat"
#   2) 在 4.SAC/2.distillation 目录下运行本脚本
#      python train_student_distill_o5_multihead_V2.py
#
# 功能：
#   - 从 distill_out 下不同的蒸馏 CSV 中读取样本
#   - 训练 5 维多头学生网络 (o5)，输入 [mu,var,H2,CL_t,step_t]
#   - 标准情况 (full / wo_rich / wo_traj) 采用 QAT 损失：
#         L1( tanh(head_a) * DELTA_CL_MAX , ΔCL_teacher )
#   - w/o QAT (post-quant) 模式下，直接拟合 ΔCL_teacher：
#         L1( head_a , ΔCL_teacher )
#   - 输出：权重 pt、归一化 JSON、训练报告 JSON

import os
import json
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =======================
# 训练模式 & 固定配置
# =======================

# 可选: "full", "wo_rich", "wo_traj", "wo_qat"
MODE            = "wo_qat"

OUT_DIR         = "distill_out"
DELTA_CL_MAX    = 2.0          # 硬件动作最大幅度，一定要和 SAC / FPGA 对齐
SEED            = 42
VAL_SPLIT       = 0.15         # 验证集比例
MAX_EPOCHS      = 400
LR              = 1e-3
WEIGHT_DECAY    = 1e-5
PATIENCE        = 40           # 早停耐心（epoch）
MIN_VAL_SAMPLES = 64           # 验证集最少样本数

if MODE == "full":
    CSV_PATH = os.path.join(OUT_DIR, "distill_data.csv")
    TAG      = "o5_multihead"
    W_NAME   = "student_o5_multihead.pt"
    J_NAME   = "obs_norm_o5_multihead.json"
    RPT_NAME = "train_report_o5_multihead_V2.json"
elif MODE == "wo_rich":
    CSV_PATH = os.path.join(OUT_DIR, "distill_data_wo_rich.csv")
    TAG      = "o5_multihead_wo_rich"
    W_NAME   = "student_wo_rich.pt"
    J_NAME   = "obs_norm_wo_rich.json"
    RPT_NAME = "train_report_wo_rich.json"
elif MODE == "wo_traj":
    CSV_PATH = os.path.join(OUT_DIR, "distill_data_wo_traj.csv")
    TAG      = "o5_multihead_wo_traj"
    W_NAME   = "student_wo_traj.pt"
    J_NAME   = "obs_norm_wo_traj.json"
    RPT_NAME = "train_report_wo_traj.json"
elif MODE == "wo_qat":
    # 与 full 共享同一份蒸馏数据，只是损失函数不同
    CSV_PATH = os.path.join(OUT_DIR, "distill_data.csv")
    TAG      = "o5_multihead_wo_qat"
    W_NAME   = "student_wo_qat.pt"
    J_NAME   = "obs_norm_wo_qat.json"
    RPT_NAME = "train_report_wo_qat.json"
else:
    raise ValueError(f"未知 MODE: {MODE}")


# =======================
# 工具函数
# =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到蒸馏样本 CSV：{path}")

    df = pd.read_csv(path)
    needed = ["mu", "var", "H2", "CL_t", "step_t", "a_teacher"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(
            f"CSV 缺少字段：{miss}\n"
            f"请确认 collect_distill_dual_V2.py 写入字段与本脚本一致。"
        )

    # 清洗 NaN / Inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=needed)
    if len(df) == 0:
        raise ValueError("CSV 经过清洗后没有有效样本。")
    return df


def build_xy_o5(df: pd.DataFrame):
    """构造 o5 学生的 (X, y)。"""
    X = df[["mu", "var", "H2", "CL_t", "step_t"]].values.astype(np.float32)
    y = df["a_teacher"].values.astype(np.float32)[:, None]
    return X, y


def compute_norm(X: np.ndarray):
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std


# =======================
# 网络结构
# =======================
class MLP5(nn.Module):
    def __init__(self, in_dim: int, h1: int = 128, h2: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(inplace=True),
            nn.Linear(h1, h2), nn.ReLU(inplace=True),
        )
        self.head_a   = nn.Linear(h2, 1)
        self.head_aux = nn.Linear(h2, 1)  # 预留辅助头，不参与损失

    def forward(self, x):
        z = self.trunk(x)
        out_a   = self.head_a(z)
        out_aux = self.head_aux(z)
        return out_a, out_aux


# =======================
# 训练单个学生
# =======================
def train_one_o5(Xn, y_hw, in_dim, device="cpu"):
    """训练单个 o5 学生。"""
    set_seed(SEED)

    N = Xn.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)

    # 验证集数量
    if N > MIN_VAL_SAMPLES * 2:
        n_val = max(int(N * VAL_SPLIT), MIN_VAL_SAMPLES)
    else:
        n_val = max(int(N * 0.2), 1)

    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    Xtr = torch.tensor(Xn[tr_idx], dtype=torch.float32, device=device)
    ytr = torch.tensor(y_hw[tr_idx], dtype=torch.float32, device=device)
    Xva = torch.tensor(Xn[val_idx], dtype=torch.float32, device=device)
    yva = torch.tensor(y_hw[val_idx], dtype=torch.float32, device=device)

    net = MLP5(in_dim=in_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    l1  = nn.L1Loss()

    best_state = None
    best_val_mae = 1e9
    bad = 0

    print(f"==> MODE = {MODE} | 训练样本 = {len(tr_idx)}, 验证样本 = {len(val_idx)}")

    for ep in range(1, MAX_EPOCHS + 1):
        net.train()
        opt.zero_grad()

        out_a, _ = net(Xtr)   # raw 输出

        if MODE == "wo_qat":
            # 不做量化感知，直接拟合 ΔCL_teacher（硬件动作域）
            pred = out_a
            loss = l1(pred, ytr)
        else:
            # QAT 模式：通过 tanh * DELTA_CL_MAX 映射到硬件动作域
            pred_hw = torch.tanh(out_a) * DELTA_CL_MAX
            loss = l1(pred_hw, ytr)

        loss.backward()
        opt.step()

        # 验证
        net.eval()
        with torch.no_grad():
            va_a, _ = net(Xva)
            if MODE == "wo_qat":
                va_pred = va_a
            else:
                va_pred = torch.tanh(va_a) * DELTA_CL_MAX
            val_mae = l1(va_pred, yva).item()

        if val_mae + 1e-6 < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep % 20 == 0 or ep == 1:
            print(f"[Ep {ep:03d}] loss={loss.item():.6f} | val_MAE={val_mae:.6f} | best={best_val_mae:.6f} | bad={bad}")

        if bad >= PATIENCE:
            print(f"早停触发（连续 {PATIENCE} 轮未提升），停止训练。")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    else:
        print("警告：没有找到更优验证集，使用最后一轮权重。")

    return net, best_val_mae


# =======================
# 保存模型与归一化
# =======================
def save_model_and_norm(net, mean, std, out_dir, tag, w_name, j_name):
    ensure_dir(out_dir)

    w_path = os.path.join(out_dir, w_name)
    torch.save(net.state_dict(), w_path)

    j_path = os.path.join(out_dir, j_name)
    norm_info = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "delta_cl_max": float(DELTA_CL_MAX),
        "mode": MODE,
        "tag": tag,
    }
    with open(j_path, "w", encoding="utf-8") as f:
        json.dump(norm_info, f, indent=2, ensure_ascii=False)

    return w_path, j_path


# =======================
# 主流程
# =======================
def main():
    print(f"==> 学生训练脚本启动, MODE = {MODE}")
    print("==> 读取蒸馏样本 CSV:", CSV_PATH)

    ensure_dir(OUT_DIR)
    df = load_csv(CSV_PATH)
    print(f"样本总数: {len(df)}")

    X, y = build_xy_o5(df)         # X: [N,5], y: [N,1]
    mean, std = compute_norm(X)
    Xn = apply_norm(X, mean, std)
    y_hw = y

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==> 使用设备: {device}")
    print("==> 开始训练 o5 学生 (5D, 多头) ...")

    net, best_mae = train_one_o5(Xn, y_hw, in_dim=Xn.shape[1], device=device)

    # 保存模型 & 归一化
    w_path, j_path = save_model_and_norm(net, mean, std, OUT_DIR, TAG, W_NAME, J_NAME)

    # 训练报告
    rpt = {
        "mode": MODE,
        "tag": TAG,
        "num_samples": int(len(X)),
        "val_MAE_abs": float(best_mae),
        "csv_path": os.path.abspath(CSV_PATH),
        "weights": os.path.abspath(w_path),
        "norm_json": os.path.abspath(j_path),
        "delta_cl_max": float(DELTA_CL_MAX),
    }
    rpt_path = os.path.join(OUT_DIR, RPT_NAME)
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump(rpt, f, indent=2, ensure_ascii=False)

    print("\n==> 训练完成。")
    print(f"[{MODE}] 验证 MAE_abs(硬件域) = {best_mae:.6f} | 样本数 = {len(X)}")
    print(f"[{MODE}] 已保存权重: {w_path}")
    print(f"[{MODE}] 已保存归一化参数: {j_path}")
    print(f"[{MODE}] 训练报告: {rpt_path}")


if __name__ == "__main__":
    main()
