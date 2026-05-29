# -*- coding: utf-8 -*-
"""
verify_student_5step_hw_like.py

目的：
- 用“和 FPGA 完全同构”的 Q4.12 逻辑做 5 步 ΔCL 闭环仿真：
    obs5_q12 -> 学生 MLP(Q4.12) -> y_raw_q12
             -> tanh_hw_q12(y_raw_q12)
             -> ΔCL_q12 = tanh * 2.0
             -> CL_next_q12 = sat16(CL + ΔCL_q12)
  重复 5 次，得到 final_cl_q12。

- 用的 MLP 权重直接从 student_o5_multihead.pt 量化为 Q4.12
  （和你之前 verify_student_export_from_files.py 的 reference 分支完全一致，
   已经验证过和 SVH/HEX 一致）

这样得到的 5 步闭环结果，应该和 FPGA actor5_ctrl 的日志对得上。
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn

# =================== 路径配置（和你之前脚本保持一致） ===================
ROOT = os.path.dirname(os.path.abspath(__file__))
STUDENT_WEIGHTS = os.path.join(ROOT, "distill_out", "student_o5_multihead.pt")
STUDENT_NORM    = os.path.join(ROOT, "distill_out", "obs_norm_o5_multihead.json")

# =================== Q4.12 基本配置 ===================
BITS = 16
FRAC = 12
Q_ONE = 1 << FRAC
Q_MIN = -(1 << (BITS - 1))
Q_MAX = (1 << (BITS - 1)) - 1

DELTA_CL_MAX = 2.0
DELTA_CL_MAX_Q12 = int(round(DELTA_CL_MAX * Q_ONE))  # 8192

def to_q12_int16(arr: np.ndarray) -> np.ndarray:
    x = np.round(arr * Q_ONE).astype(np.int64)
    x = np.clip(x, Q_MIN, Q_MAX).astype(np.int16)
    return x

def q12_to_float(x: int) -> float:
    return float(x) / Q_ONE

def sat16(x: int) -> int:
    if x > Q_MAX:
        return Q_MAX
    if x < Q_MIN:
        return Q_MIN
    return int(x)

def relu_q12(x: int) -> int:
    return 0 if x < 0 else x

def q12_mul(a: int, b: int) -> int:
    """Q4.12 * Q4.12 -> Q4.12（饱和）"""
    t = int(a) * int(b)      # Q8.24
    y = t >> FRAC            # 回 Q4.12
    return sat16(y)

# =================== tanh_Q12：和 SV PWL 完全同构 ===================
def tanh_q12_hw(x_q12: int) -> int:
    """
    SystemVerilog 版（你 actor5_ctrl 里那段）的一模一样翻译：
        breakpoints: 0.5, 1.5, 3.0, 4.0  (Q4.12：2048, 6144, 12288, 16384)
        分段线性系数：
            M1 = 3784
            M2 = 1815, B2 = 985
            M3 = 245,  B3 = 3342
            M4 = 18,   B4 = 4020
        输出 |tanh| ≤ 1.0 → Q4.12 上限 4096
    全部用整数模拟，避免任何浮点误差。
    """
    # 常量（直接照抄 SV）
    BP_0p5 = 2048   # 0.5 * 4096
    BP_1p5 = 6144   # 1.5 * 4096
    BP_3p0 = 12288  # 3.0 * 4096
    BP_4p0 = 16384  # 4.0 * 4096

    M1 = 3784
    M2 = 1815
    B2 = 985
    M3 = 245
    B3 = 3342
    M4 = 18
    B4 = 4020

    ONE_Q12 = 4096

    x = int(x_q12)
    sign_neg = (x < 0)
    ax = -x if sign_neg else x  # |x|

    if ax <= BP_0p5:
        y = (M1 * ax) >> FRAC
    elif ax <= BP_1p5:
        y = ((M2 * ax) >> FRAC) + B2
    elif ax <= BP_3p0:
        y = ((M3 * ax) >> FRAC) + B3
    elif ax <= BP_4p0:
        y = ((M4 * ax) >> FRAC) + B4
    else:
        y = ONE_Q12

    y = int(y)
    if sign_neg:
        y = -y
    # 再做一次饱和，防止越界
    return sat16(y)

# =================== 学生网络结构（和训练一致） ===================
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

# =================== 从 PyTorch 直接量化出 Q4.12 权重 ===================
def get_student_q12_from_torch():
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

    return W1_q, b1_q, W2_q, b2_q, W3_q, b3_q

# =================== 单层 Q4.12 前向（和之前 verify 脚本一致） ===================
def simulate_layer_q12(input_vec_q12, W_q12, b_q12, apply_relu=True):
    """
    input_vec_q12: [in_dim] int16 (Q4.12)
    W_q12:         [out_dim, in_dim] int16 (Q4.12)
    b_q12:         [out_dim] int16 (Q4.12)
    """
    input_vec_q12 = np.asarray(input_vec_q12, dtype=np.int64)
    out_dim, in_dim = W_q12.shape
    assert input_vec_q12.shape[0] == in_dim

    outputs = []
    for i in range(out_dim):
        acc_q24 = np.int64(0)
        for j in range(in_dim):
            prod_q24 = np.int64(W_q12[i, j]) * np.int64(input_vec_q12[j])  # Q8.24
            acc_q24 += prod_q24
        acc_q24 += (np.int64(b_q12[i]) << FRAC)  # bias Q4.12 -> Q8.24
        y_q12 = int(acc_q24 >> FRAC)             # back to Q4.12
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

# =================== 5-step 闭环仿真（硬件等价） ===================
def simulate_5step_hw_like(obs5_init_q12, W1, b1, W2, b2, W3, b3, steps=5):
    """
    obs5_init_q12: 初始 5 维观测（Q4.12，int16）
        [mu, var, H2, CL_norm, step_norm]

    CL 更新规则：
        每步：
            obs[3] = 当前 CL
            y_raw_q12 = MLP(obs)                # Q4.12
            tanh_q12  = tanh_q12_hw(y_raw_q12)  # Q4.12
            dCL_q12   = q12_mul(tanh_q12, DELTA_CL_MAX_Q12)  # Q4.12
            cl_next   = sat16(CL + dCL_q12)
    """
    obs5_init_q12 = np.asarray(obs5_init_q12, dtype=np.int16)
    assert obs5_init_q12.shape[0] == 5

    mu_q12   = int(obs5_init_q12[0])
    var_q12  = int(obs5_init_q12[1])
    H2_q12   = int(obs5_init_q12[2])
    cl_q12   = int(obs5_init_q12[3])
    step_q12 = int(obs5_init_q12[4])

    print("=== 5-step HW-like simulation ===")
    print(f"init obs5_q12 = {obs5_init_q12.tolist()}")
    print("init obs5_float ≈", [round(q12_to_float(v), 6) for v in obs5_init_q12])
    print()

    for k in range(steps):
        # 组当前步的 obs（和 FPGA 一样，只更新 CL 这一维）
        obs_k = np.array([mu_q12, var_q12, H2_q12, cl_q12, step_q12], dtype=np.int16)

        y_raw_q12 = forward_mlp_q12(obs_k, W1, b1, W2, b2, W3, b3)
        y_raw_f   = q12_to_float(y_raw_q12)

        tanh_q    = tanh_q12_hw(y_raw_q12)
        tanh_f    = q12_to_float(tanh_q)

        dcl_q12   = q12_mul(tanh_q, DELTA_CL_MAX_Q12)
        dcl_f     = q12_to_float(dcl_q12)

        cl_sum    = cl_q12 + dcl_q12
        cl_next   = sat16(cl_sum)
        cl_next_f = q12_to_float(cl_next)

        print(f"[step {k}]")
        print(f"  CL_in_q12   = {cl_q12:6d}  (≈ {q12_to_float(cl_q12): .6f})")
        print(f"  y_raw_q12   = {y_raw_q12:6d}  (≈ {y_raw_f: .6f})")
        print(f"  tanh_q12    = {tanh_q:6d}  (≈ {tanh_f: .6f})")
        print(f"  dCL_q12     = {dcl_q12:6d}  (≈ {dcl_f: .6f})")
        print(f"  CL_sum_q12  = {cl_sum:6d}")
        print(f"  CL_next_q12 = {cl_next:6d}  (≈ {cl_next_f: .6f})")
        print()

        cl_q12 = cl_next  # 进入下一步

    print(f"final CL_q12 = {cl_q12}  (≈ {q12_to_float(cl_q12): .6f})")
    return cl_q12

# =================== main ===================
def main():
    print("========== 学生模型 5-step HW-like 仿真 ==========")
    W1, b1, W2, b2, W3, b3 = get_student_q12_from_torch()

    # 这里用你之前那组 obs5_q12
    obs5_q12 = np.array([
        -7452,   # mu
         892,    # var
        -2307,   # H2
         963,    # CL_t
        -5792    # step_t
    ], dtype=np.int16)

    final_cl_q12 = simulate_5step_hw_like(
        obs5_init_q12=obs5_q12,
        W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
        steps=5
    )

    print("\n[结果] final_cl_q12 =", final_cl_q12,
          "(float ≈", q12_to_float(final_cl_q12), ")")

if __name__ == "__main__":
    main()
