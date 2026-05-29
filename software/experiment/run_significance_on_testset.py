# -*- coding: utf-8 -*-
"""
run_significance_on_testset.py (v2)
No-arg, hard-coded search. Produces:
1) Paired tests (Wilcoxon + paired t + bootstrap CI) on:
   - ALL
   - by_scene (if scene column exists)
   - hard-case subsets pre-defined (top-K worst by Input / by Baseline)
   - union/intersection of hard-cases to avoid cherry-picking
2) Tail robustness table (p90/p95/worst-10 mean) with bootstrap CI + permutation test.

Outputs:
  - metrics_testset_significance.csv
  - metrics_testset_significance.tex
  - metrics_testset_tail_robustness.csv
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats


# =========================
# Fixed settings (NO CLI)
# =========================
SEED = 2025
N_BOOT = 10000
N_PERM = 20000              # permutation tests for tail stats
WRITE_BY_SCENE = True

DETAIL_CSV_NAME = "metrics_testset_detail.csv"

# hard-case definition (pre-registered)
HARDCASE_FRAC = 0.20         # top 20% worst
TAIL_PCTS = [90, 95]         # p90, p95
WORST_FRAC = 0.10            # worst-10% mean

SEARCH_DIRS = [
    ".", "./test", "..", "../test", "../experiment", "../experiment/test",
    "../../", "../../test",
]

COL_METHOD_CANDIDATES = ["method", "Method", "algo", "Algorithm", "name"]
COL_SCENE_CANDIDATES  = ["scene", "Scene", "category", "Category"]
COL_IMAGE_CANDIDATES  = ["image", "Image", "fname", "filename", "path", "img", "stem"]

# metric aliases
METRIC_CANON = {
    "Q_BRISQUE": ["Q_BRISQUE", "BRISQUE", "brisque", "q_brisque"],
    "Q_NIQE":    ["Q_NIQE", "NIQE", "niqe", "q_niqe"],
}

# direction
LOWER_BETTER = {"Q_BRISQUE": True, "Q_NIQE": True}

# method match rules
METHOD_A_REGEX = r"(proposed|dt[\s\-–]?qat|dual[\s\-–]?teacher|student)"
METHOD_B_REGEX = r"(fixed).*?(clahe).*?(cl\s*=\s*2(\.0)?)"
METHOD_INPUT_REGEX = r"(^input$|^\s*input\s*$|input\b)"


# =========================
# Utils
# =========================
def find_existing_file(base_dir: Path, filename: str, rel_dirs: List[str]) -> Path:
    for d in rel_dirs:
        p = (base_dir / d / filename).resolve()
        if p.exists() and p.is_file():
            return p
    hits = list(base_dir.rglob(filename))
    if hits:
        return hits[0].resolve()
    raise FileNotFoundError(f"Cannot find {filename} under {base_dir}")

def pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_method_series(s: pd.Series) -> pd.Series:
    return s.astype(str).fillna("").str.strip()

def auto_pick_method(methods: List[str], pattern: str, allow_none=False) -> Optional[str]:
    rx = re.compile(pattern, re.IGNORECASE)
    matches = [m for m in methods if rx.search(m)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return sorted(matches, key=len, reverse=True)[0]
    if allow_none:
        return None
    raise ValueError(
        f"Cannot auto-pick method by pattern: {pattern}\n"
        f"Available methods({len(methods)}): {methods[:40]}{'...' if len(methods)>40 else ''}\n"
        f"Fix regex constants in this script."
    )

def auto_find_metric_cols(df: pd.DataFrame) -> Dict[str, str]:
    cols = set(df.columns)
    found = {}
    for canon, aliases in METRIC_CANON.items():
        for a in aliases:
            if a in cols:
                found[canon] = a
                break
    missing = [k for k in METRIC_CANON.keys() if k not in found]
    if missing:
        raise KeyError(f"Missing metric columns for: {missing}. Existing columns: {list(df.columns)[:60]}")
    return found

def fmt(x, nd=4):
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}"

def bootstrap_ci(values: np.ndarray, func, n_boot: int, seed: int) -> Tuple[float, float]:
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samp = values[idx]
    boot = np.apply_along_axis(func, 1, samp)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)

def paired_stats(a: np.ndarray, b: np.ndarray, lower_better: bool, n_boot: int, seed: int) -> Dict:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask].astype(np.float64)
    b = b[mask].astype(np.float64)
    n = int(a.size)
    if n == 0:
        return {"n": 0}

    d = a - b

    win_rate = float(np.mean(a < b)) if lower_better else float(np.mean(a > b))

    # Wilcoxon
    try:
        p_w2 = float(stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
        p_w1 = float(stats.wilcoxon(d, alternative=("less" if lower_better else "greater"),
                                    zero_method="wilcox").pvalue)
    except Exception:
        p_w2, p_w1 = np.nan, np.nan

    # paired t-test
    try:
        p_t2 = float(stats.ttest_rel(a, b, nan_policy="omit").pvalue)
    except Exception:
        p_t2 = np.nan

    # effect sizes
    pos = np.sum(d > 0)
    neg = np.sum(d < 0)
    cliff = float((pos - neg) / n)

    sd = float(np.std(d, ddof=1)) if n > 1 else np.nan
    dz = float(np.mean(d) / sd) if (np.isfinite(sd) and sd > 0) else np.nan

    # bootstrap on d
    ci_mean = bootstrap_ci(d, np.mean, n_boot, seed)
    ci_med  = bootstrap_ci(d, np.median, n_boot, seed + 17)

    return {
        "n": n,
        "mean_A": float(np.mean(a)), "mean_B": float(np.mean(b)),
        "mean_diff_A_minus_B": float(np.mean(d)),
        "median_diff_A_minus_B": float(np.median(d)),
        "win_rate_A_better": win_rate,
        "cliff_delta_paired": cliff,
        "cohen_dz": dz,
        "wilcoxon_p_two_sided": p_w2,
        "wilcoxon_p_one_sided": p_w1,
        "ttest_p_two_sided": p_t2,
        "boot_ci_mean_lo": float(ci_mean[0]),
        "boot_ci_mean_hi": float(ci_mean[1]),
        "boot_ci_median_lo": float(ci_med[0]),
        "boot_ci_median_hi": float(ci_med[1]),
    }

def tail_stats(values: np.ndarray, lower_better: bool) -> Dict:
    v = values[np.isfinite(values)].astype(np.float64)
    n = int(v.size)
    if n == 0:
        return {"n": 0}
    # worst tail = largest values when lower is better
    sort_v = np.sort(v)
    if lower_better:
        worst = sort_v[::-1]
    else:
        worst = sort_v  # if higher better, worst are small
    k = max(1, int(np.ceil(WORST_FRAC * n)))
    worst_mean = float(np.mean(worst[:k]))
    out = {"n": n, "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if n > 1 else np.nan,
           "worst_mean": worst_mean}
    for p in TAIL_PCTS:
        out[f"p{p}"] = float(np.percentile(v, p))
    return out

def bootstrap_tail_diff(a: np.ndarray, b: np.ndarray, lower_better: bool, n_boot: int, seed: int) -> Dict:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask].astype(np.float64)
    b = b[mask].astype(np.float64)
    n = int(a.size)
    if n == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    a_s = a[idx]; b_s = b[idx]

    def stat_pack(x):
        st = tail_stats(x, lower_better)
        return st

    # compute diffs for each bootstrap
    diffs = { "mean": [], "worst_mean": [] }
    for p in TAIL_PCTS:
        diffs[f"p{p}"] = []

    for i in range(n_boot):
        sa = stat_pack(a_s[i])
        sb = stat_pack(b_s[i])
        diffs["mean"].append(sa["mean"] - sb["mean"])
        diffs["worst_mean"].append(sa["worst_mean"] - sb["worst_mean"])
        for p in TAIL_PCTS:
            diffs[f"p{p}"].append(sa[f"p{p}"] - sb[f"p{p}"])

    out = {"n": n}
    for k, arr in diffs.items():
        arr = np.array(arr, dtype=np.float64)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[f"diff_{k}"] = float(np.mean(arr))
        out[f"ci_{k}_lo"] = float(lo)
        out[f"ci_{k}_hi"] = float(hi)
    return out

def permutation_pvalue_tail(a: np.ndarray, b: np.ndarray, lower_better: bool, stat_key: str,
                            n_perm: int, seed: int) -> float:
    """Paired permutation test on tail statistic difference (A-B)."""
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask].astype(np.float64)
    b = b[mask].astype(np.float64)
    n = int(a.size)
    if n == 0:
        return np.nan
    rng = np.random.default_rng(seed)

    def stat(x):
        st = tail_stats(x, lower_better)
        return st[stat_key]

    obs = stat(a) - stat(b)

    # random sign-flips on paired differences
    # equivalent to swapping pairs
    diffs = []
    for _ in range(n_perm):
        flip = rng.random(n) < 0.5
        aa = a.copy(); bb = b.copy()
        aa[flip], bb[flip] = bb[flip], aa[flip]
        diffs.append(stat(aa) - stat(bb))
    diffs = np.array(diffs, dtype=np.float64)
    # two-sided p-value
    p = float((np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n_perm + 1))
    return p

def build_wide(df: pd.DataFrame, col_method: str, key_cols: List[str], metric_colmap: Dict[str, str]) -> pd.DataFrame:
    sub = df.copy()
    sub["_pair_id"] = sub[key_cols].astype(str).agg("/".join, axis=1)
    wide = pd.DataFrame({"_pair_id": sorted(sub["_pair_id"].unique())}).set_index("_pair_id")
    for canon, real_col in metric_colmap.items():
        piv = sub.pivot_table(index="_pair_id", columns=col_method, values=real_col, aggfunc="first")
        # join all methods columns
        piv = piv.add_prefix(f"{canon}__")
        wide = wide.join(piv, how="left")
    # bring scene if exists as first element in key (often scene/image)
    wide = wide.reset_index(drop=False)
    if len(key_cols) >= 2:
        wide["scene"] = wide["_pair_id"].astype(str).str.split("/", n=1).str[0]
    return wide

def pick_hardcase_ids(wide: pd.DataFrame, col: str, frac: float, lower_better: bool) -> List[str]:
    """Pick top frac worst based on reference column col."""
    s = wide.set_index("_pair_id")[col]
    s = s[np.isfinite(s)]
    if s.empty:
        return []
    n = s.size
    k = max(1, int(np.ceil(frac * n)))
    # worst = largest if lower better, else smallest
    s_sorted = s.sort_values(ascending=not lower_better)
    return s_sorted.index[:k].tolist()


# =========================
# Main
# =========================
def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = find_existing_file(base_dir, DETAIL_CSV_NAME, SEARCH_DIRS)
    print(f"[FOUND] detail csv: {csv_path}")

    df = pd.read_csv(csv_path)

    col_method = pick_first_existing_col(df, COL_METHOD_CANDIDATES)
    if col_method is None:
        raise KeyError(f"Cannot find method column in {df.columns.tolist()}")
    df[col_method] = normalize_method_series(df[col_method])

    col_scene = pick_first_existing_col(df, COL_SCENE_CANDIDATES)
    col_image = pick_first_existing_col(df, COL_IMAGE_CANDIDATES)
    if col_image is None:
        raise KeyError("Cannot find image-like column for pairing (image/path/filename).")

    # key columns: prefer scene+image if scene exists
    key_cols = [c for c in [col_scene, col_image] if c is not None]
    metric_colmap = auto_find_metric_cols(df)

    methods = sorted([m for m in df[col_method].unique() if str(m).strip() != ""])
    method_a = auto_pick_method(methods, METHOD_A_REGEX)
    method_b = auto_pick_method(methods, METHOD_B_REGEX)
    method_in = auto_pick_method(methods, METHOD_INPUT_REGEX, allow_none=True)

    print(f"[PICK] Method A: {method_a}")
    print(f"[PICK] Method B: {method_b}")
    print(f"[PICK] Input   : {method_in if method_in else '(not found)'}")

    # build wide table for all methods/metrics
    wide = build_wide(df, col_method, key_cols, metric_colmap)

    # groups (predefined, to avoid cherry picking)
    groups = [("ALL", wide["_pair_id"].tolist())]

    # by_scene (not cherry picking: report all scenes)
    if WRITE_BY_SCENE and "scene" in wide.columns:
        for sc, g in wide.groupby("scene"):
            ids = g["_pair_id"].tolist()
            if len(ids) >= 5:
                groups.append((f"SCENE::{sc}", ids))

    # hard-case subsets based on Input / Baseline worst
    # NOTE: selection uses reference metric Q_BRISQUE by default (primary in your paper)
    primary_metric = "Q_BRISQUE"
    lb = LOWER_BETTER[primary_metric]

    col_in_ref = f"{primary_metric}__{method_in}" if method_in else None
    col_b_ref  = f"{primary_metric}__{method_b}"

    hard_in = pick_hardcase_ids(wide, col_in_ref, HARDCASE_FRAC, lb) if (col_in_ref and col_in_ref in wide.columns) else []
    hard_b  = pick_hardcase_ids(wide, col_b_ref,  HARDCASE_FRAC, lb) if (col_b_ref in wide.columns) else []

    # also union / intersection to make it robust against critique
    set_in = set(hard_in)
    set_b  = set(hard_b)
    hard_u = sorted(list(set_in.union(set_b)))
    hard_i = sorted(list(set_in.intersection(set_b)))

    if hard_in:
        groups.append((f"HARD::{primary_metric}::InputTop{int(HARDCASE_FRAC*100)}", hard_in))
    if hard_b:
        groups.append((f"HARD::{primary_metric}::BaseTop{int(HARDCASE_FRAC*100)}", hard_b))
    if hard_u:
        groups.append((f"HARD::{primary_metric}::UnionTop{int(HARDCASE_FRAC*100)}", hard_u))
    if hard_i:
        groups.append((f"HARD::{primary_metric}::InterTop{int(HARDCASE_FRAC*100)}", hard_i))

    # ===== Paired tests output =====
    sig_rows = []
    for grp_name, ids in groups:
        g = wide[wide["_pair_id"].isin(ids)].copy()
        for canon_metric in ["Q_BRISQUE", "Q_NIQE"]:
            col_a = f"{canon_metric}__{method_a}"
            col_b = f"{canon_metric}__{method_b}"
            if col_a not in g.columns or col_b not in g.columns:
                continue
            a = g[col_a].to_numpy()
            b = g[col_b].to_numpy()
            res = paired_stats(a, b, LOWER_BETTER[canon_metric],
                              n_boot=(max(2000, N_BOOT // 5) if grp_name != "ALL" else N_BOOT),
                              seed=(SEED + (0 if grp_name == "ALL" else 97)))
            if res.get("n", 0) == 0:
                continue
            sig_rows.append({
                "group": grp_name,
                "metric": canon_metric,
                "method_A": method_a,
                "method_B": method_b,
                **res
            })

    sig_df = pd.DataFrame(sig_rows)
    out_sig_csv = csv_path.parent / "metrics_testset_significance.csv"
    sig_df.to_csv(out_sig_csv, index=False)
    print(f"[OK] wrote: {out_sig_csv}")

    # ===== Tail robustness output =====
    tail_rows = []
    for grp_name, ids in groups:
        g = wide[wide["_pair_id"].isin(ids)].copy()
        for canon_metric in ["Q_BRISQUE", "Q_NIQE"]:
            col_a = f"{canon_metric}__{method_a}"
            col_b = f"{canon_metric}__{method_b}"
            if col_a not in g.columns or col_b not in g.columns:
                continue
            a = g[col_a].to_numpy()
            b = g[col_b].to_numpy()

            # raw tail stats per method
            st_a = tail_stats(a, LOWER_BETTER[canon_metric])
            st_b = tail_stats(b, LOWER_BETTER[canon_metric])
            if st_a.get("n", 0) == 0:
                continue

            # bootstrap diff CI
            bt = bootstrap_tail_diff(a, b, LOWER_BETTER[canon_metric],
                                     n_boot=(max(4000, N_BOOT) if grp_name == "ALL" else max(2000, N_BOOT // 2)),
                                     seed=SEED + 123)

            # permutation p-values for key tail stats
            pvals = {}
            for key in (["mean", "worst_mean"] + [f"p{p}" for p in TAIL_PCTS]):
                pvals[f"perm_p_{key}"] = permutation_pvalue_tail(
                    a, b, LOWER_BETTER[canon_metric], key, n_perm=(N_PERM if grp_name == "ALL" else max(5000, N_PERM // 2)),
                    seed=SEED + 777
                )

            row = {
                "group": grp_name,
                "metric": canon_metric,
                "n": st_a["n"],
                "A_mean": st_a["mean"], "B_mean": st_b["mean"],
                "A_std": st_a["std"], "B_std": st_b["std"],
                "A_worst_mean": st_a["worst_mean"], "B_worst_mean": st_b["worst_mean"],
                **{f"A_p{p}": st_a[f"p{p}"] for p in TAIL_PCTS},
                **{f"B_p{p}": st_b[f"p{p}"] for p in TAIL_PCTS},
                **bt,
                **pvals,
                "method_A": method_a,
                "method_B": method_b,
                "hardcase_frac": HARDCASE_FRAC,
                "worst_frac": WORST_FRAC,
            }
            tail_rows.append(row)

    tail_df = pd.DataFrame(tail_rows)
    out_tail_csv = csv_path.parent / "metrics_testset_tail_robustness.csv"
    tail_df.to_csv(out_tail_csv, index=False)
    print(f"[OK] wrote: {out_tail_csv}")

    # ===== Minimal TeX lines (ALL + HARD groups only) =====
    out_tex = csv_path.parent / "metrics_testset_significance.tex"
    lines = []
    lines.append("% Auto-generated. Columns: group | metric | mean(A-B) | 95% CI | Wilcoxon p(two-sided) | win-rate")
    keep_groups = ["ALL"] + [g for g, _ in groups if g.startswith("HARD::")]
    for grp in keep_groups:
        for canon_metric in ["Q_BRISQUE", "Q_NIQE"]:
            r = sig_df[(sig_df["group"] == grp) & (sig_df["metric"] == canon_metric)]
            if r.empty:
                continue
            r = r.iloc[0]
            ci = f"[{r['boot_ci_mean_lo']:.3f},{r['boot_ci_mean_hi']:.3f}]"
            lines.append(
                f"{grp} & {canon_metric} & {r['mean_diff_A_minus_B']:.3f} & {ci} & {r['wilcoxon_p_two_sided']:.3g} & {r['win_rate_A_better']:.2f} \\\\"
            )
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] wrote: {out_tex}")

    # ===== Console summary =====
    print("\n================ Summary (ALL) ================")
    for canon_metric in ["Q_BRISQUE", "Q_NIQE"]:
        r = sig_df[(sig_df["group"] == "ALL") & (sig_df["metric"] == canon_metric)]
        if r.empty:
            continue
        r = r.iloc[0]
        print(f"\nMetric={canon_metric}, n={int(r['n'])}")
        print(f"  mean(A)={fmt(r['mean_A'])}  mean(B)={fmt(r['mean_B'])}")
        print(f"  mean diff(A-B)={fmt(r['mean_diff_A_minus_B'])}  95% CI[{fmt(r['boot_ci_mean_lo'])},{fmt(r['boot_ci_mean_hi'])}]")
        print(f"  Wilcoxon p(two-sided)={fmt(r['wilcoxon_p_two_sided'],6)}  win-rate={fmt(r['win_rate_A_better'],3)}")

    print("\n================ Summary (HARD) ================")
    for grp in [g for g, _ in groups if g.startswith("HARD::")]:
        rB = sig_df[(sig_df["group"] == grp) & (sig_df["metric"] == "Q_BRISQUE")]
        if rB.empty:
            continue
        rB = rB.iloc[0]
        print(f"\n[{grp}] BRISQUE n={int(rB['n'])}: diff={rB['mean_diff_A_minus_B']:.3f}, CI=[{rB['boot_ci_mean_lo']:.3f},{rB['boot_ci_mean_hi']:.3f}], p={rB['wilcoxon_p_two_sided']:.3g}, win={rB['win_rate_A_better']:.2f}")

if __name__ == "__main__":
    main()
