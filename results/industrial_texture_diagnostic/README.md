# Industrial Texture Diagnostic

This directory contains derived metrics for the high-texture industrial-scene
diagnostic used in the TECS revision response.

## Source Boundary

The local diagnostic used a 72-image balanced sample from a GitHub mirror of
NEU steel-surface defect images:

`https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection`

The source industrial images are **not** redistributed in this repository. This
directory only includes derived CSV summaries and a metric-only plot.

## Files

- `method_summary_overall.csv`: overall averages across 72 images.
- `method_summary_by_class.csv`: per-class averages for six defect classes.
- `industrial_texture_metric_bars.png`: plot generated from summary metrics;
  it does not contain source images.

## Key Interpretation

The diagnostic should be used as limitation evidence rather than as a universal
win claim.

On the 72-image sample:

- Fixed CLAHE with OpenCV `CL=2.0` is the most conservative enhancement
  baseline among the CLAHE variants.
- Rule-based adaptive CLAHE selects a large average clip limit and strongly
  amplifies high-frequency texture/noise.
- The proposed DT-QAT controller is less aggressive than the rule-based
  adaptive baseline, but still more aggressive than Fixed CLAHE `CL=2.0` on
  this out-of-distribution industrial sample.

This supports a conservative response to the reviewer: highly textured or noisy
industrial surfaces remain a boundary case for the current clip-limit-only
controller, especially because industrial defect images were not included in
the training/distillation data.
