# -*- coding: utf-8 -*-
"""
run_ablation_scenewise_summary.py

功能：
- 读取 metrics_ablation_students.csv
- 从 image 字段中解析场景名（image 形如 "低照度20\\dense_61.jpg"）
- 按 (scene, config) 分组，计算 Q_BRISQUE / Q_NIQE 的场景平均
- 打印：
    1）场景 × 配置的数值表（方便你自己看）
    2）一份 LaTeX 表格代码（默认展示四个配置）
    3）一份只对比 "w/o QAT (post-quant)" 和 "Full (DT--QAT)" 的精简版 LaTeX 表
"""

import os
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT_DIR, "metrics_ablation_students.csv")


def parse_scene(image_rel_path: str) -> str:
    """
    从 image 相对路径中提取场景名：
      - "低照度20\\dense_61.jpg" -> "低照度20"
      - "雾霾7/494.png" -> "雾霾7"
      - "img.png" -> "root"
    """
    # 统一分隔符
    p = image_rel_path.replace("\\", "/")
    parts = p.split("/")
    if len(parts) == 1:
        return "root"
    return parts[0]


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"未找到 CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    required_cols = ["image", "config", "Q_BRISQUE", "Q_NIQE"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")

    # 解析场景名
    df["scene"] = df["image"].astype(str).apply(parse_scene)

    # 分组求平均
    grp = df.groupby(["scene", "config"], as_index=False)[["Q_BRISQUE", "Q_NIQE"]].mean()

    # 排个序：先按场景名，再按配置名
    grp = grp.sort_values(["scene", "config"]).reset_index(drop=True)

    print("===== 场景 × 配置 平均 BRISQUE / NIQE =====")
    print(grp.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ---------------- LaTeX 表格 1：四配置完整版 ----------------
    print("\n===== LaTeX 表格：所有配置（可直接粘贴到论文） =====\n")

    # 场景列表 & 配置列表（按你现在的命名）
    scenes = sorted(df["scene"].unique())
    configs = [
        "w/o Rich teacher",
        "w/o trajectory selection",
        "w/o QAT (post-quant)",
        "Full (DT--QAT)",
    ]

    # 做个透视方便查数：索引 (scene, config)
    pivot = grp.set_index(["scene", "config"])

    # 生成 LaTeX 代码
    print(r"\begin{table}[!t]")
    print(r"    \centering")
    print(r"    \caption{Scene-wise ablation results on BRISQUE / NIQE}")
    print(r"    \label{tab:ablation_scenewise}")
    print(r"    \begin{tabular}{lcccc}")
    print(r"        \toprule")
    print(r"        Scene & w/o Rich & w/o Traj. & w/o QAT & Full (DT--QAT) \\")
    print(r"        \midrule")

    for scene in scenes:
        row_vals = []
        for cfg in configs:
            if (scene, cfg) in pivot.index:
                b = pivot.loc[(scene, cfg), "Q_BRISQUE"]
                n = pivot.loc[(scene, cfg), "Q_NIQE"]
                row_vals.append(f"{b:.2f}/{n:.2f}")
            else:
                row_vals.append("--")
        # 注意：有中文场景名时，LaTeX 建议用 \text{...} 包一下或直接使用 CJK 宏包
        print(f"        {scene} & " + " & ".join(row_vals) + r" \\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")

    # ---------------- LaTeX 表格 2：只看 QAT 影响（可选） ----------------
    print("\n===== LaTeX 表格：只对比 w/o QAT 和 Full（可选） =====\n")

    print(r"\begin{table}[!t]")
    print(r"    \centering")
    print(r"    \caption{Scene-wise comparison of w/o QAT vs. Full (DT--QAT)}")
    print(r"    \label{tab:ablation_scene_qat}")
    print(r"    \begin{tabular}{lcc}")
    print(r"        \toprule")
    print(r"        Scene & w/o QAT (post-quant) & Full (DT--QAT) \\")
    print(r"        \midrule")

    for scene in scenes:
        # w/o QAT
        if (scene, "w/o QAT (post-quant)") in pivot.index:
            b_q = pivot.loc[(scene, "w/o QAT (post-quant)"), "Q_BRISQUE"]
            n_q = pivot.loc[(scene, "w/o QAT (post-quant)"), "Q_NIQE"]
            val_q = f"{b_q:.2f}/{n_q:.2f}"
        else:
            val_q = "--"

        # Full
        if (scene, "Full (DT--QAT)") in pivot.index:
            b_f = pivot.loc[(scene, "Full (DT--QAT)"), "Q_BRISQUE"]
            n_f = pivot.loc[(scene, "Full (DT--QAT)"), "Q_NIQE"]
            val_f = f"{b_f:.2f}/{n_f:.2f}"
        else:
            val_f = "--"

        print(f"        {scene} & {val_q} & {val_f} \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
