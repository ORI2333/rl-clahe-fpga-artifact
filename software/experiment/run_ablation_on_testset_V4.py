# -*- coding: utf-8 -*-
"""
run_ablation_on_testset_V4.py

功能：
- 读取 run_ablation_on_testset_V3.py 生成的逐图明细:
    metrics_ablation_students.csv
  （列: image, config, CL, Q_BRISQUE, Q_NIQE）

- 计算每个 ablation 配置的分布/尾部稳健性统计：
    * BRISQUE: mean, std, p90, p95, worst-10% mean
    * NIQE:    mean, std, worst-10% mean

- 输出：
    * metrics_ablation_tail_stats.csv   : 方便你复制数值进论文
    * metrics_ablation_tail_stats.tex   : 直接可粘贴进 LaTeX（含完整 table 环境）

路径策略（与 V3/V4 指标脚本保持一致）：
- ROOT_DIR = 当前脚本所在目录（通常为 4.SAC/experiment）
- 默认读取:  ROOT_DIR/metrics_ablation_students.csv
- 默认输出:  ROOT_DIR/metrics_ablation_tail_stats.csv
            ROOT_DIR/metrics_ablation_tail_stats.tex
"""

import os
import csv
from typing import Dict, List, Tuple, Any

import numpy as np

# ----------------- 基本路径配置（默认无需改） -----------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DETAIL_CSV = os.path.join(ROOT_DIR, "metrics_ablation_students.csv")
OUT_TAIL_CSV = os.path.join(ROOT_DIR, "metrics_ablation_tail_stats.csv")
OUT_TEX = os.path.join(ROOT_DIR, "metrics_ablation_tail_stats.tex")

# 论文表格建议的行顺序（若你 V3 的 config 名称一致，将自动按此顺序输出）
CONFIG_ORDER = [
    "w/o Rich teacher",
    "w/o trajectory selection",
    "w/o QAT (post-quant)",
    "Full (DT--QAT)",
]

# ----------------- 统计函数 -----------------
def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def summarize_array(x: List[float]) -> Dict[str, float]:
    """
    返回：mean, std, p90, p95, worst10
    worst10: 取“最差 10%（分数最大）”的均值；若样本太少，至少取 1 个。
    """
    arr = np.array([v for v in x if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return dict(mean=np.nan, std=np.nan, p90=np.nan, p95=np.nan, worst10=np.nan, n=0)

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))

    # percentiles
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))

    # worst 10% mean
    k = max(1, int(np.ceil(arr.size * 0.10)))
    worst = np.sort(arr)[-k:]
    worst10 = float(np.mean(worst))

    return dict(mean=mean, std=std, p90=p90, p95=p95, worst10=worst10, n=int(arr.size))

def fmt_pm(mean: float, std: float, nd: int = 2) -> str:
    """LaTeX: $mean\pm std$（保留 nd 位小数）。"""
    if not (np.isfinite(mean) and np.isfinite(std)):
        return r"--"
    return rf"${mean:.{nd}f}\pm{std:.{nd}f}$"

def fmt_num(x: float, nd: int = 2) -> str:
    if not np.isfinite(x):
        return r"--"
    return f"{x:.{nd}f}"

# ----------------- 读取与聚合 -----------------
def read_detail_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到明细 CSV：{path}\n请先运行 run_ablation_on_testset_V3.py 生成 metrics_ablation_students.csv")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    if not rows:
        raise RuntimeError(f"明细 CSV 为空：{path}")
    required = {"image", "config", "Q_BRISQUE", "Q_NIQE"}
    if not required.issubset(set(rows[0].keys())):
        raise RuntimeError(f"明细 CSV 缺少必要列 {sorted(required)}，实际列={list(rows[0].keys())}")
    return rows

def aggregate_by_config(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, List[float]]]:
    agg: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        cfg = (r.get("config") or "").strip()
        if cfg == "":
            continue
        qb = _safe_float(r.get("Q_BRISQUE"))
        qn = _safe_float(r.get("Q_NIQE"))
        agg.setdefault(cfg, {"brisque": [], "niqe": []})
        agg[cfg]["brisque"].append(qb)
        agg[cfg]["niqe"].append(qn)
    return agg

# ----------------- 输出 -----------------
def write_tail_csv(out_path: str, table: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = [
        "config",
        "QB_mean", "QB_std", "QB_p90", "QB_p95", "QB_worst10",
        "QN_mean", "QN_std", "QN_worst10",
        "N",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in table:
            w.writerow(row)

def write_tex_table(out_path: str, rows_tex: List[str]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tex = []
    tex.append(r"\begin{table}[width=.98\linewidth,cols=7,pos=h]")
    tex.append(r"\caption{Distributional and tail-risk analysis for ablations on the test set. "
               r"We report mean$\pm$std, upper-tail percentiles ($p90/p95$), and worst-10\% mean. "
               r"Lower is better for BRISQUE/NIQE.}")
    tex.append(r"\label{tab:ablation_tail}")
    tex.append(r"\small")
    tex.append(r"\setlength{\tabcolsep}{3.5pt}")
    tex.append(r"\begin{tabular*}{\tblwidth}{@{} LCCCCCC@{} }")
    tex.append(r"\toprule")
    tex.append(r"Config & \makecell{$Q_B$\\mean$\pm$std} & \makecell{$Q_B$\\$p90$} & "
               r"\makecell{$Q_B$\\$p95$} & \makecell{$Q_B$\\worst10\%} & "
               r"\makecell{$Q_N$\\mean$\pm$std} & \makecell{$Q_N$\\worst10\%} \\")
    tex.append(r"\midrule")
    tex.extend(rows_tex)
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular*}")
    tex.append(r"\end{table}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")

def main():
    print("========== Ablation Tail/Robustness Summary (V4) ==========")
    print(f"[信息] 当前脚本目录: {ROOT_DIR}")
    print(f"[信息] 读取明细 CSV: {DETAIL_CSV}")

    rows = read_detail_csv(DETAIL_CSV)
    agg = aggregate_by_config(rows)

    # 输出顺序：优先按 CONFIG_ORDER，其余配置按名称补在后面
    all_cfgs = list(agg.keys())
    ordered = [c for c in CONFIG_ORDER if c in agg]
    extras = sorted([c for c in all_cfgs if c not in ordered])
    cfg_list = ordered + extras

    table_rows: List[Dict[str, Any]] = []
    tex_rows: List[str] = []

    for cfg in cfg_list:
        qb = summarize_array(agg[cfg]["brisque"])
        qn = summarize_array(agg[cfg]["niqe"])

        row = dict(
            config=cfg,
            QB_mean=qb["mean"], QB_std=qb["std"], QB_p90=qb["p90"], QB_p95=qb["p95"], QB_worst10=qb["worst10"],
            QN_mean=qn["mean"], QN_std=qn["std"], QN_worst10=qn["worst10"],
            N=qb["n"],
        )
        table_rows.append(row)

        tex_line = " & ".join([
            cfg,
            fmt_pm(qb["mean"], qb["std"]),
            fmt_num(qb["p90"]),
            fmt_num(qb["p95"]),
            fmt_num(qb["worst10"]),
            fmt_pm(qn["mean"], qn["std"]),
            fmt_num(qn["worst10"]),
        ]) + r" \\"
        tex_rows.append(tex_line)

    # 写 CSV
    print(f"[信息] 写入尾部统计 CSV: {OUT_TAIL_CSV}")
    write_tail_csv(OUT_TAIL_CSV, table_rows)

    # 写 TEX（完整 table 环境）
    print(f"[信息] 写入 LaTeX 表格: {OUT_TEX}")
    write_tex_table(OUT_TEX, tex_rows)

    print("\n----- LaTeX rows (copy/paste) -----")
    for line in tex_rows:
        print(line)
    print("---------- Done ----------")

if __name__ == "__main__":
    main()
