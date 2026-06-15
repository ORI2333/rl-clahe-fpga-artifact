# Mask-Aware MVTec Diagnostic

This directory contains derived diagnostic evidence from a small MVTec AD
texture subset used to strengthen the high-texture industrial-scene response.

## Source Boundary

The local run used 20 anomalous samples from the Hugging Face
`Voxel51/mvtec-ad` conversion of MVTec AD:

- `carpet`: 10 color-defect samples
- `grid`: 10 glue-defect samples

MVTec AD provides pixel-precise anomaly masks and is released under
CC BY-NC-SA 4.0. Source images and masks are **not** redistributed in this
artifact. This directory contains only derived metric summaries and a derived
representative panel.

## Files

- `masked_summary_overall.csv`: method-level mask-aware diagnostic means.
- `masked_summary_by_class.csv`: class-level means for carpet and grid.
- `masked_summary_by_defect.csv`: defect-type means.
- `analysis_summary.md`: Markdown summary produced by the local script.
- `masked_representative_panel.png`: derived visual panel for qualitative
  inspection.

## Metrics

The diagnostic uses ground-truth masks to compute:

- defect/background contrast gain,
- contrast-to-noise ratio (CNR) gain,
- background and defect high-frequency gain,
- high-frequency selectivity,
- simple saliency AUROC and best Dice.

The saliency AUROC/Dice values come from a simple contrast/gradient saliency
map, not from a trained detector. Therefore this is mask-aware diagnostic
evidence, not detector-level deployment performance.

## Key Interpretation

On this 20-image texture subset, rule-based adaptive CLAHE increases
defect/background contrast the most, but also produces the largest background
high-frequency amplification and reduces CNR. The proposed controller is more
conservative: it gives smaller contrast gain but keeps CNR closer to the input
and amplifies background high-frequency content less than the more aggressive
baselines.

This supports the reviewer-facing interpretation that high clip-limit adaptive
CLAHE can improve apparent defect contrast while also amplifying structured
industrial background texture. Stronger edge/high-frequency response should not
be equated with better industrial detection accuracy without detector-level
validation.
