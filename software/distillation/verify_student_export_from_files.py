# -*- coding: utf-8 -*-
"""
verify_student_export_from_files.py

目的：
- 验证 export_student_q12_multihead_V2.py 导出的 SVH / HEX 文件是否正确。
- 对比：
    1) 直接从 PyTorch 模型量化得到的 Q4.12 权重 (reference)
    2) 从 layer1_weights.svh / l2_w_b0.hex / l3_w.hex 等文件中读回的 Q4.12 权重 (from_file)
- 在同一个 obs5_q12 输入下，做全定点前向，比较两边输出 y_q12 和 ΔCL 是否一致。

前置条件：
- 已经运行过 export_student_q12_multihead_V2.py，生成对应的 SVH 和 HEX 文件。
"""

import os
import re
import json
import math
import numpy as np
import torch
import torch.nn as nn

# =============== 路径配置（与 export 脚本保持一致） ===============
ROOT = os.path.dirname(os.path.abspath(__file__))

OUT_SVH = os.path.join(ROOT, "../FPGA", "rtl", "verilog_headers")
OUT_HEX = os.path.join(ROOT, "../FPGA", "hex")

STUDENT_WEIGHTS = os.path.join(ROOT, "distill_out", "student_o5_multihead.pt")
STUDENT_NORM    = os.path.join(ROOT, "distill_out", "obs_norm_o5_multihead.json")

# =============== Q4.12 配置与工具函数 ===============
BITS = 16
FRAC = 12
Q_MIN = -(1 << (BITS - 1))
Q_MAX = (1 << (BITS - 1)) - 1
Q_ONE = 1 << FRAC

def to_q12_int16(arr: np.ndarray) -> np.ndarray:
    """浮点 -> Q4.12 int16（四舍五入 + 饱和）"""
    x = np.round(arr * Q_ONE).astype(np.int64)
    x = np.clip(x, Q_MIN, Q_MAX).astype(np.int16)
    return x

def q12_to_float(x: int) -> float:
    return float(x) / Q_ONE

def sat16(x: int) -> int:
    return int(max(Q_MIN, min(Q_MAX, x)))

def relu_q12(x: int) -> int:
    return 0 if x < 0 else x

def hex_to_signed16(h: str) -> int:
    """4位16进制 -> 有符号int16"""
    v = int(h, 16)
    if v >= 0x8000:
        v -= 0x10000
    return v

# =============== 学生网络结构（与训练脚本一致） ===============
class MLP5(nn.Module):
    def __init__(self, in_dim, h1=128, h2=64):
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

# =============== 定点前向 ===============
def simulate_layer_q12(input_vec_q12, W_q12, b_q12, apply_relu=True):
    """
    input_vec_q12: [in_dim] int16 (Q4.12)
    W_q12: [out_dim, in_dim] int16 (Q4.12)
    b_q12: [out_dim] int16 (Q4.12)
    """
    input_vec_q12 = np.asarray(input_vec_q12, dtype=np.int64)
    out_dim, in_dim = W_q12.shape
    assert input_vec_q12.shape[0] == in_dim, f"输入维度 {input_vec_q12.shape[0]} != {in_dim}"

    outputs = []
    for i in range(out_dim):
        acc_q24 = np.int64(0)
        for j in range(in_dim):
            prod_q24 = np.int64(W_q12[i, j]) * np.int64(input_vec_q12[j])  # Q4.12*Q4.12=Q8.24
            acc_q24 += prod_q24
        acc_q24 += (np.int64(b_q12[i]) << FRAC)  # 偏置 Q4.12 -> Q8.24
        y_q12 = int(acc_q24 >> FRAC)  # 回到 Q4.12
        y_q12 = sat16(y_q12)
        if apply_relu:
            y_q12 = relu_q12(y_q12)
        outputs.append(y_q12)

    return np.array(outputs, dtype=np.int16)

def forward_mlp_q12(obs5_q12, W1, b1, W2, b2, W3, b3):
    """
    三层 MLP 定点前向：
      x -> L1(ReLU) -> L2(ReLU) -> L3(no ReLU)
    """
    x = np.asarray(obs5_q12, dtype=np.int16)
    x = simulate_layer_q12(x, W1, b1, apply_relu=True)   # 5 -> 128
    x = simulate_layer_q12(x, W2, b2, apply_relu=True)   # 128 -> 64
    x = simulate_layer_q12(x, W3, b3, apply_relu=False)  # 64 -> 1
    assert x.shape[0] == 1
    return int(x[0])

# =============== 从 PyTorch 直接量化（reference） ===============
def get_reference_q12_from_torch():
    if not os.path.exists(STUDENT_WEIGHTS):
        raise FileNotFoundError(f"找不到学生权重: {STUDENT_WEIGHTS}")
    if not os.path.exists(STUDENT_NORM):
        raise FileNotFoundError(f"找不到学生归一化 JSON: {STUDENT_NORM}")

    with open(STUDENT_NORM, "r", encoding="utf-8") as f:
        nj = json.load(f)
    mean = np.array(nj["mean"], dtype=np.float32)
    in_dim = mean.shape[0]

    net = MLP5(in_dim=in_dim)
    state = torch.load(STUDENT_WEIGHTS, map_location="cpu")
    net.load_state_dict(state, strict=True)
    net.eval()

    l1 = net.trunk[0]
    l2 = net.trunk[2]
    l3 = net.head_a

    W1_f = l1.weight.detach().cpu().numpy()  # [h1, in_dim]
    b1_f = l1.bias.detach().cpu().numpy()    # [h1]
    W2_f = l2.weight.detach().cpu().numpy()  # [h2, h1]
    b2_f = l2.bias.detach().cpu().numpy()    # [h2]
    W3_f = l3.weight.detach().cpu().numpy()  # [1, h2]
    b3_f = l3.bias.detach().cpu().numpy()    # [1]

    W1_q = to_q12_int16(W1_f)
    b1_q = to_q12_int16(b1_f)
    W2_q = to_q12_int16(W2_f)
    b2_q = to_q12_int16(b2_f)
    W3_q = to_q12_int16(W3_f)
    b3_q = to_q12_int16(b3_f)

    # W3_q 变成 [1,h2] -> [1,64]
    return W1_q, b1_q, W2_q, b2_q, W3_q, b3_q

# =============== 从文件读回 L1（SVH） ===============
def load_l1_from_svh():
    wp = os.path.join(OUT_SVH, "layer1_weights.svh")
    bp = os.path.join(OUT_SVH, "layer1_biases.svh")

    if not os.path.exists(wp) or not os.path.exists(bp):
        raise FileNotFoundError("缺少 L1 SVH 文件 layer1_weights.svh 或 layer1_biases.svh")

    # 先解析 LAYER1_IN_DIM / LAYER1_OUT_DIM
    in_dim = None
    out_dim = None
    with open(wp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("localparam int LAYER1_IN_DIM"):
                m = re.search(r"=\s*(\d+)\s*;", line)
                if m:
                    in_dim = int(m.group(1))
            elif line.startswith("localparam int LAYER1_OUT_DIM"):
                m = re.search(r"=\s*(\d+)\s*;", line)
                if m:
                    out_dim = int(m.group(1))
    if in_dim is None or out_dim is None:
        raise RuntimeError("未能从 layer1_weights.svh 解析出 LAYER1_IN_DIM / OUT_DIM")

    # 解析所有 16'shXXXX
    weight_vals = []
    with open(wp, "r", encoding="utf-8") as f:
        for line in f:
            for m in re.finditer(r"16'sh([0-9a-fA-F]{4})", line):
                h = m.group(1)
                weight_vals.append(hex_to_signed16(h))

    if len(weight_vals) != in_dim * out_dim:
        raise RuntimeError(
            f"L1 权重数量不匹配，期望 {out_dim}*{in_dim}={out_dim*in_dim}，实际 {len(weight_vals)}"
        )

    W1 = np.array(weight_vals, dtype=np.int16).reshape(out_dim, in_dim)

    # 解析偏置
    bias_vals = []
    with open(bp, "r", encoding="utf-8") as f:
        for line in f:
            for m in re.finditer(r"16'sh([0-9a-fA-F]{4})", line):
                h = m.group(1)
                bias_vals.append(hex_to_signed16(h))
    if len(bias_vals) != out_dim:
        raise RuntimeError(
            f"L1 偏置数量不匹配，期望 {out_dim}，实际 {len(bias_vals)}"
        )

    b1 = np.array(bias_vals, dtype=np.int16)

    print(f"[L1] 从 SVH 读回: in_dim={in_dim}, out_dim={out_dim}")
    return W1, b1

# =============== 从 HEX 读回 L2 ===============
def load_l2_from_hex(out_dim, in_dim, P=64, prefix="l2"):
    """
    读回：
      - {prefix}_w_b0.hex
      - {prefix}_b_b0.hex
    说明：export 时写法为：
      对每个 in_idx：
        words = [W2[out_idx, in_idx] for out_idx in batch范围内] 然后 reversed 再写一行
    这里需要反向还原回 W2[out_dim, in_dim]。
    """
    num_batches = (out_dim + P - 1) // P

    W2 = np.zeros((out_dim, in_dim), dtype=np.int16)
    b2 = np.zeros((out_dim,), dtype=np.int16)

    for batch in range(num_batches):
        start = batch * P
        pw = os.path.join(OUT_HEX, f"{prefix}_w_b{batch}.hex")
        pb = os.path.join(OUT_HEX, f"{prefix}_b_b{batch}.hex")

        if not os.path.exists(pw) or not os.path.exists(pb):
            raise FileNotFoundError(f"缺少 {pw} 或 {pb}")

        # ---- 读权重 ----
        with open(pw, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) != in_dim:
            raise RuntimeError(
                f"[L2] 权重行数 != in_dim，期望 {in_dim}，实际 {len(lines)}"
            )

        for in_idx, line in enumerate(lines):
            if len(line) % 4 != 0:
                raise RuntimeError(f"[L2] 第 {in_idx} 行长度非法: {len(line)}")
            words_rev = [line[i:i+4] for i in range(0, len(line), 4)]
            P_actual = len(words_rev)
            for rev_i, h in enumerate(words_rev):
                out_idx = start + (P_actual - 1 - rev_i)  # 对应 export 中 reversed
                if out_idx < out_dim:
                    W2[out_idx, in_idx] = hex_to_signed16(h)

        # ---- 读偏置 ----
        with open(pb, "r") as f:
            first_line = None
            for line in f:
                line = line.strip()
                if line:
                    first_line = line
                    break
        if first_line is None:
            raise RuntimeError(f"[L2] 偏置文件 {pb} 为空")

        if len(first_line) % 4 != 0:
            raise RuntimeError(f"[L2] 偏置行长度非法: {len(first_line)}")
        words_rev = [first_line[i:i+4] for i in range(0, len(first_line), 4)]
        P_actual = len(words_rev)
        for rev_i, h in enumerate(words_rev):
            out_idx = start + (P_actual - 1 - rev_i)
            if out_idx < out_dim:
                b2[out_idx] = hex_to_signed16(h)

    print(f"[L2] 从 HEX 读回: in_dim={in_dim}, out_dim={out_dim}, num_batches={num_batches}")
    return W2, b2

# =============== 从 HEX 读回 L3 ===============
def load_l3_from_hex(h2):
    pw = os.path.join(OUT_HEX, "l3_w.hex")
    pb = os.path.join(OUT_HEX, "l3_b.hex")

    if not os.path.exists(pw) or not os.path.exists(pb):
        raise FileNotFoundError("缺少 l3_w.hex 或 l3_b.hex")

    w_vals = []
    with open(pw, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            w_vals.append(hex_to_signed16(line))
    if len(w_vals) != h2:
        raise RuntimeError(
            f"[L3] 权重数量不匹配，期望 {h2}，实际 {len(w_vals)}"
        )
    W3 = np.array(w_vals, dtype=np.int16).reshape(1, h2)

    with open(pb, "r") as f:
        b_line = None
        for line in f:
            line = line.strip()
            if line:
                b_line = line
                break
    if b_line is None:
        raise RuntimeError("l3_b.hex 为空")
    b3 = np.array([hex_to_signed16(b_line)], dtype=np.int16)

    print(f"[L3] 从 HEX 读回: h2={h2}")
    return W3, b3

# =============== 检查 obs_normalizer.svh ===============
def verify_obs_normalizer():
    svh_path = os.path.join(OUT_SVH, "obs_normalizer.svh")
    if not os.path.exists(svh_path):
        print("[Norm] 未找到 obs_normalizer.svh，跳过检查。")
        return

    with open(STUDENT_NORM, "r", encoding="utf-8") as f:
        nj = json.load(f)
    mean = np.array(nj["mean"], dtype=np.float64)
    std  = np.array(nj["std"],  dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)

    s_ref = 1.0 / std
    b_ref = -mean * s_ref

    s_q_ref = to_q12_int16(s_ref)
    b_q_ref = to_q12_int16(b_ref)

    s_vals = []
    b_vals = []

    with open(svh_path, "r", encoding="utf-8") as f:
        for line in f:
            if "OBS_NORM_S" in line or "OBS_NORM_B" in line:
                # 后续行才是常量
                continue
            for m in re.finditer(r"16'sh([0-9a-fA-F]{4})", line):
                h = m.group(1)
                if "OBS_NORM_S" in line:
                    s_vals.append(hex_to_signed16(h))
                elif "OBS_NORM_B" in line:
                    b_vals.append(hex_to_signed16(h))

    # 保险起见简单一点：再扫一遍分别抓
    s_vals, b_vals = [], []
    with open(svh_path, "r", encoding="utf-8") as f:
        mode = None
        for line in f:
            line_strip = line.strip()
            if line_strip.startswith("localparam logic signed [15:0] OBS_NORM_S"):
                mode = "S"
                continue
            if line_strip.startswith("localparam logic signed [15:0] OBS_NORM_B"):
                mode = "B"
                continue
            if mode in ("S", "B"):
                for m in re.finditer(r"16'sh([0-9a-fA-F]{4})", line):
                    h = m.group(1)
                    if mode == "S":
                        s_vals.append(hex_to_signed16(h))
                    else:
                        b_vals.append(hex_to_signed16(h))

    s_vals = np.array(s_vals, dtype=np.int16)
    b_vals = np.array(b_vals, dtype=np.int16)

    if s_vals.shape[0] != s_q_ref.shape[0] or b_vals.shape[0] != b_q_ref.shape[0]:
        print("[Norm] 维度不匹配，无法精确对比。")
        return

    diff_s = s_vals.astype(np.int32) - s_q_ref.astype(np.int32)
    diff_b = b_vals.astype(np.int32) - b_q_ref.astype(np.int32)

    print("\n[Norm] obs_normalizer.svh 检查结果：")
    print("  S_from_file =", s_vals)
    print("  S_ref_q12   =", s_q_ref)
    print("  diff_S      =", diff_s)
    print("  B_from_file =", b_vals)
    print("  B_ref_q12   =", b_q_ref)
    print("  diff_B      =", diff_b)

# =============== 主流程 ===============
def main():
    print("========== 验证学生模型导出的 SVH/HEX ==========")

    # 1. 从 PyTorch 直接量化 (reference)
    W1_ref, b1_ref, W2_ref, b2_ref, W3_ref, b3_ref = get_reference_q12_from_torch()
    h1, in_dim = W1_ref.shape
    h2 = W2_ref.shape[0]
    print(f"[Ref] 结构: in_dim={in_dim}, h1={h1}, h2={h2}")

    # 2. 从文件读回 (from_file)
    W1_f, b1_f = load_l1_from_svh()
    W2_f, b2_f = load_l2_from_hex(out_dim=h2, in_dim=h1, P=64, prefix="l2")
    W3_f, b3_f = load_l3_from_hex(h2=h2)

    # 3. 给定 obs5_q12，做两套前向
    obs5_q12 = np.array([
        -7452,   # obs5[0]
         892,    # obs5[1]
        -2307,   # obs5[2]
         963,    # obs5[3]
        -5792    # obs5[4]
    ], dtype=np.int16)

    print("\n[输入] obs5_q12 =", obs5_q12)
    print("       obs5_float ≈", [round(q12_to_float(v), 6) for v in obs5_q12])

    # reference
    y_ref_q12 = forward_mlp_q12(obs5_q12, W1_ref, b1_ref, W2_ref, b2_ref, W3_ref, b3_ref)
    y_ref_float = q12_to_float(y_ref_q12)

    # from_file
    y_file_q12 = forward_mlp_q12(obs5_q12, W1_f, b1_f, W2_f, b2_f, W3_f, b3_f)
    y_file_float = q12_to_float(y_file_q12)

    # ΔCL（用 delta_cl_max=2.0）
    delta_cl_max = 2.0
    dcl_ref = math.tanh(y_ref_float) * delta_cl_max
    dcl_file = math.tanh(y_file_float) * delta_cl_max

    print("\n------ 前向结果对比 ------")
    print(f"[Ref ] y_q12 = {y_ref_q12:6d}, y_float ≈ {y_ref_float:.6f}, ΔCL ≈ {dcl_ref:.6f}")
    print(f"[File] y_q12 = {y_file_q12:6d}, y_float ≈ {y_file_float:.6f}, ΔCL ≈ {dcl_file:.6f}")
    print(f"差值: Δy_q12 = {y_file_q12 - y_ref_q12}, Δy_float ≈ {y_file_float - y_ref_float:.8f}, ΔΔCL ≈ {dcl_file - dcl_ref:.8f}")

    # 4. 检查 obs_normalizer.svh（可选）
    verify_obs_normalizer()

if __name__ == "__main__":
    main()
