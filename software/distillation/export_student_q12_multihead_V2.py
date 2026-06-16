# -*- coding: utf-8 -*-
"""
export_student_q12_multihead_V2.py

从「多头学生网络」(student_o5_multihead.pt) 与 obs_norm_o5_multihead.json 中
导出 FPGA 侧需要的 Q4.12 权重/偏置/归一化参数。

学生结构（与 train_student_distill_o5_multihead_* 保持一致）：
    输入维度 in_dim = 5  （[mu, var, H2, CL_t, step_t]）
    trunk:   Linear(in_dim, 128) -> ReLU -> Linear(128, 64) -> ReLU
    head_a:  Linear(64, 1)       # 只导出这一头，作为 ΔCL 预测
    head_aux: Linear(64, 1)      # 忽略，不导出

输出文件（命名风格尽量与旧 export_fpga_assets_q12.py 保持一致）：
    ../rtl/verilog_headers/layer1_weights.svh   (Q4.12, int16, [H1][IN_DIM] = [128][5])
    ../rtl/verilog_headers/layer1_biases.svh
    ../rtl/verilog_headers/obs_normalizer.svh   (Q4.12, 5 维)

    ../hex/l2_w_b0.hex    # 第二层 128→64，按 64 宽度打包，只生成一个 batch
    ../hex/l2_b_b0.hex
    ../hex/l3_w.hex       # 输出层 64→1
    ../hex/l3_b.hex

量化格式：Q4.12，int16（-32768..32767），FRAC=12
"""

import os
import json
import sys
import numpy as np
import torch
import torch.nn as nn

# ===================== 路径配置 =====================
ROOT = os.path.dirname(os.path.abspath(__file__))

OUT_SVH = os.path.join(ROOT, "../FPGA", "rtl", "verilog_headers")
OUT_HEX = os.path.join(ROOT, "../FPGA", "hex")
os.makedirs(OUT_SVH, exist_ok=True)
os.makedirs(OUT_HEX, exist_ok=True)

STUDENT_WEIGHTS = os.path.join(ROOT, "distill_out", "student_o5_multihead.pt")
STUDENT_NORM    = os.path.join(ROOT, "distill_out", "obs_norm_o5_multihead.json")

# ===================== 量化配置 =====================
BITS = 16
FRAC = 12
P    = 64   # 宽 ROM 批宽（64×16bit）

Q_MIN = -(1 << (BITS - 1))
Q_MAX = (1 << (BITS - 1)) - 1

# ===================== 学生网络结构（与训练脚本一致） =====================
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
        self.head_a = nn.Linear(h2, 1)   # ΔCL 头
        self.head_aux = nn.Linear(h2, 1) # 其它辅助头（这里不导出）

    def forward(self, x):
        z = self.trunk(x)
        a = self.head_a(z)
        aux = self.head_aux(z)
        return a, aux

# ===================== 工具函数 =====================
def to_q12_int16(arr: np.ndarray) -> np.ndarray:
    """浮点 -> Q4.12 int16（四舍五入 + 饱和）"""
    x = np.round(arr * (1 << FRAC)).astype(np.int64)
    x = np.clip(x, Q_MIN, Q_MAX).astype(np.int16)
    return x

def check_nonzero_hex(path: str):
    """快速检查 hex 是否为全 0（仅打印提示，不影响功能）"""
    if not os.path.isfile(path):
        print(f"[检查] {path} 不存在")
        return
    try:
        with open(path, "r") as f:
            data = f.read().strip().replace("\n", "")
        if len(data) == 0:
            print(f"[检查] {path}: 空文件")
            return
        nz = any(ch not in "0 \t\r" for ch in data)
        print(f"[检查] {path}: {'OK(非零)' if nz else '全0!!!'}")
    except Exception as e:
        print(f"[检查] {path}: 读取失败 {e}")

# ===================== 写出：L1（in_dim→h1）到 SVH =====================
def write_l1_svh_student(W_q12: np.ndarray, b_q12: np.ndarray):
    """
    生成：
    - layer1_weights.svh : localparam logic signed [15:0] LAYER1_WEIGHTS [H1][IN_DIM]
    - layer1_biases.svh  : localparam logic signed [15:0] LAYER1_BIASES  [H1]
    这里 H1 = W_q12.shape[0]，IN_DIM = W_q12.shape[1]
    """
    h1, in_dim = W_q12.shape
    if b_q12.shape[0] != h1:
        raise ValueError(f"[L1] 偏置长度应为 {h1}，实际 {b_q12.shape[0]}")

    wp = os.path.join(OUT_SVH, "layer1_weights.svh")
    bp = os.path.join(OUT_SVH, "layer1_biases.svh")

    with open(wp, "w", encoding="utf-8") as f:
        f.write("// 自动生成：Student L1 权重 (Q4.12, int16)\n")
        f.write(f"localparam int LAYER1_IN_DIM  = {in_dim};\n")
        f.write(f"localparam int LAYER1_OUT_DIM = {h1};\n")
        f.write("localparam logic signed [15:0] LAYER1_WEIGHTS ")
        f.write("[LAYER1_OUT_DIM-1:0][LAYER1_IN_DIM-1:0] = '{\n")
        for i in range(h1):
            elems = ", ".join(
                f"16'sh{int(W_q12[i, j]) & 0xFFFF:04x}" for j in range(in_dim)
            )
            f.write(f"    '{{ {elems} }}{',' if i < h1-1 else ''}\n")
        f.write("};\n")
    print("写出:", wp)

    with open(bp, "w", encoding="utf-8") as f:
        f.write("// 自动生成：Student L1 偏置 (Q4.12, int16)\n")
        f.write("localparam logic signed [15:0] LAYER1_BIASES [LAYER1_OUT_DIM-1:0] = '{\n")
        for i in range(h1):
            f.write(f"    16'sh{int(b_q12[i]) & 0xFFFF:04x}{',' if i < h1-1 else ''}\n")
        f.write("};\n")
    print("写出:", bp)

# ===================== 写出：L2（h1→h2）到 HEX =====================
def write_l2_hex_student(W2_q12: np.ndarray, B2_q12: np.ndarray, prefix="l2"):
    """
    第二层权重/偏置写法（参考原 L2 写法，做成通用版）：
    - out_dim = h2, in_dim = h1
    - 批宽 P=64，按输出维度分批：
        num_batches = ceil(out_dim / P)
      每个 batch 生成：
        * {prefix}_w_b{b}.hex : DEPTH = in_dim，每行拼接该 batch 内最多 64 个 out 的 16-bit（宽 64*16 = 1024 bit）
        * {prefix}_b_b{b}.hex : DEPTH = 64，写满 64 行，每行拼接该 batch 的 64 个偏置（不足的补零）
    """
    out_dim, in_dim = W2_q12.shape  # (h2, h1)
    if B2_q12.shape[0] != out_dim:
        raise ValueError(f"[L2] 偏置长度应为 {out_dim}，实际 {B2_q12.shape[0]}")

    num_batches = (out_dim + P - 1) // P

    for batch in range(num_batches):
        # 权重
        pw = os.path.join(OUT_HEX, f"{prefix}_w_b{batch}.hex")
        with open(pw, "w") as f:
            start = batch * P
            for in_idx in range(in_dim):
                words = []
                for i in range(P):
                    o = start + i
                    if o < out_dim:
                        val = int(W2_q12[o, in_idx]) & 0xFFFF
                    else:
                        val = 0
                    words.append(f"{val:04x}")
                # 反转线序，保持与原脚本一致
                f.write("".join(reversed(words)) + "\n")
        print("写出:", pw)

    for batch in range(num_batches):
        # 偏置
        pb = os.path.join(OUT_HEX, f"{prefix}_b_b{batch}.hex")
        with open(pb, "w") as f:
            start = batch * P
            line_words = []
            for i in range(P):
                o = start + i
                if o < out_dim:
                    val = int(B2_q12[o]) & 0xFFFF
                else:
                    val = 0
                line_words.append(f"{val:04x}")
            line = "".join(reversed(line_words))
            for _ in range(P):  # DEPTH=64
                f.write(line + "\n")
        print("写出:", pb)

# ===================== 写出：输出层（h2→1）到 HEX =====================
def write_l3_hex_student(W3_q12: np.ndarray, b3_q12: np.ndarray):
    """
    输出层形状：(1,h2)，写出：
      - l3_w.hex : h2 行，每行一个 16-bit
      - l3_b.hex : 1 行，一个 16-bit
    """
    if W3_q12.ndim == 2:
        if W3_q12.shape[0] == 1:
            W = W3_q12.reshape(-1)  # (1,h2) -> (h2,)
        elif W3_q12.shape[1] == 1:
            W = W3_q12.reshape(-1)  # (h2,1) 也摊平
        else:
            W = W3_q12.reshape(-1)
    else:
        W = W3_q12.reshape(-1)

    h2 = W.shape[0]

    p_w = os.path.join(OUT_HEX, "l3_w.hex")
    with open(p_w, "w") as f:
        for i in range(h2):
            f.write(f"{(int(W[i]) & 0xFFFF):04x}\n")
    print("写出:", p_w)

    p_b = os.path.join(OUT_HEX, "l3_b.hex")
    b16 = int(b3_q12.reshape(())) & 0xFFFF
    with open(p_b, "w") as f:
        f.write(f"{b16:04x}\n")
    print("写出:", p_b)

# ===================== 写出：学生 Obs Norm 到 SVH =====================
def write_student_obs_norm_svh(norm_json_path: str):
    """
    从 obs_norm_o5_multihead.json 中读取 mean/std/delta_cl_max，
    生成 obs_normalizer.svh（5 维）：
        y = (x - mean) / std ≈ s * x + b
        其中 s = 1/std, b = -mean/std
    全部使用 Q4.12。
    """
    with open(norm_json_path, "r", encoding="utf-8") as f:
        nj = json.load(f)

    mean = np.array(nj["mean"], dtype=np.float64)
    std  = np.array(nj["std"],  dtype=np.float64)

    in_dim = mean.shape[0]
    if std.shape[0] != in_dim:
        raise ValueError(f"[Norm] mean/std 维度不一致: mean={mean.shape}, std={std.shape}")

    # 防止 std 为 0
    std = np.where(std < 1e-8, 1.0, std)

    s = 1.0 / std
    b = -mean * s

    s_q = to_q12_int16(s)
    b_q = to_q12_int16(b)

    outp = os.path.join(OUT_SVH, "obs_normalizer.svh")
    with open(outp, "w", encoding="utf-8") as f:
        f.write("// 自动生成：Student 观测归一化 y = s*x + b （Q4.12）\n")
        f.write(f"localparam int OBS_NORM_DIM = {in_dim};\n")
        f.write("localparam logic signed [15:0] OBS_NORM_S [OBS_NORM_DIM-1:0] = '{\n")
        for i in range(in_dim):
            f.write(f"    16'sh{int(s_q[i]) & 0xFFFF:04x}{',' if i < in_dim-1 else ''}\n")
        f.write("};\n")
        f.write("localparam logic signed [15:0] OBS_NORM_B [OBS_NORM_DIM-1:0] = '{\n")
        for i in range(in_dim):
            f.write(f"    16'sh{int(b_q[i]) & 0xFFFF:04x}{',' if i < in_dim-1 else ''}\n")
        f.write("};\n")
    print("写出:", outp)

# ===================== 主流程 =====================
def main():
    if not os.path.exists(STUDENT_WEIGHTS):
        raise FileNotFoundError(f"未找到学生权重文件: {STUDENT_WEIGHTS}")
    if not os.path.exists(STUDENT_NORM):
        raise FileNotFoundError(f"未找到学生归一化 JSON: {STUDENT_NORM}")

    # 读取归一化，确定输入维度
    with open(STUDENT_NORM, "r", encoding="utf-8") as f:
        nj = json.load(f)
    mean = np.array(nj["mean"], dtype=np.float32)
    in_dim = mean.shape[0]
    print(f"[信息] 学生输入维度 in_dim = {in_dim}")

    # 加载学生网络
    print("[信息] 加载学生模型...")
    student = MLP5(in_dim=in_dim)
    sd = torch.load(STUDENT_WEIGHTS, map_location="cpu")
    student.load_state_dict(sd, strict=True)
    student.eval()

    # 取出三层 Linear
    l1 = student.trunk[0]   # in_dim -> h1
    l2 = student.trunk[2]   # h1 -> h2
    l3 = student.head_a     # h2 -> 1

    assert isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear) and isinstance(l3, nn.Linear)

    W1_f = l1.weight.detach().cpu().numpy()  # [h1, in_dim]
    b1_f = l1.bias.detach().cpu().numpy()    # [h1]
    W2_f = l2.weight.detach().cpu().numpy()  # [h2, h1]
    b2_f = l2.bias.detach().cpu().numpy()    # [h2]
    W3_f = l3.weight.detach().cpu().numpy()  # [1, h2]
    b3_f = l3.bias.detach().cpu().numpy()    # [1]

    h1, in_dim_check = W1_f.shape
    h2, h1_check     = W2_f.shape
    assert in_dim_check == in_dim, f"L1 in_dim={in_dim_check} 与 norm 中 {in_dim} 不一致"
    assert h1_check   == h1,       f"L2 in_dim={h1_check} 与 L1 out_dim {h1} 不一致"

    print(f"[信息] 学生结构: {in_dim} → {h1} → {h2} → 1")

    # 量化
    W1_q = to_q12_int16(W1_f)
    b1_q = to_q12_int16(b1_f)
    W2_q = to_q12_int16(W2_f)
    b2_q = to_q12_int16(b2_f)
    W3_q = to_q12_int16(W3_f)
    b3_q = to_q12_int16(b3_f)

    # 写 L1 SVH
    write_l1_svh_student(W1_q, b1_q)

    # 写 L2 HEX
    write_l2_hex_student(W2_q, b2_q, prefix="l2")

    # 写 L3 HEX
    write_l3_hex_student(W3_q, b3_q)

    # 写 Obs Norm SVH
    write_student_obs_norm_svh(STUDENT_NORM)

    # 简单检查几个 hex 文件是否非零
    for p in ["l2_w_b0.hex", "l2_b_b0.hex", "l3_w.hex", "l3_b.hex"]:
        full = os.path.join(OUT_HEX, p)
        if os.path.exists(full):
            check_nonzero_hex(full)

    print("✅ 学生模型 Q4.12 权重/偏置/归一化 导出完成。")


if __name__ == "__main__":
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(
            "usage: python software/distillation/export_student_q12_multihead_V2.py\n\n"
            "Exports Q4.12 student weights, biases, and normalization constants.\n"
            "Requires local files that are not bundled in the public artifact:\n"
            f"  {STUDENT_WEIGHTS}\n"
            f"  {STUDENT_NORM}\n"
        )
        raise SystemExit(0)
    main()
