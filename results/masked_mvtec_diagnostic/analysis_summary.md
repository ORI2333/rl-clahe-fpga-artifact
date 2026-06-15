# Mask-Aware Industrial Defect Diagnostic

## Scope
- Uses ground-truth masks to measure defect/background separability after enhancement.
- Does not train or evaluate a deployed industrial detector.
- Pixel-level saliency AUROC and best Dice are computed from a simple contrast/gradient saliency map, so they are diagnostic proxies, not detector benchmark scores.

## Overall Summary
| Method | n | CL | Contrast gain | CNR gain | Background HF gain | Defect HF gain | HF selectivity | Saliency AUROC | Best Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Input | 20 | nan | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.445 | 0.031 |
| Fixed CLAHE (CL=2.0) | 20 | 2.000 | 1.512 | 0.927 | 3.095 | 2.567 | 0.834 | 0.410 | 0.028 |
| Rule-based Adaptive CLAHE | 20 | 9.125 | 1.643 | 0.829 | 4.787 | 4.586 | 0.970 | 0.422 | 0.031 |
| Proposed DT-QAT Student | 20 | 1.171 | 1.299 | 0.951 | 2.189 | 1.832 | 0.856 | 0.420 | 0.028 |

## By-Class Summary
| Class | Method | n | Contrast gain | CNR gain | Saliency AUROC | Best Dice |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| carpet | Input | 10 | 1.000 | 1.000 | 0.282 | 0.036 |
| carpet | Fixed CLAHE (CL=2.0) | 10 | 1.398 | 0.923 | 0.210 | 0.036 |
| carpet | Rule-based Adaptive CLAHE | 10 | 1.545 | 0.750 | 0.275 | 0.036 |
| carpet | Proposed DT-QAT Student | 10 | 1.146 | 0.961 | 0.236 | 0.036 |
| grid | Input | 10 | 1.000 | 1.000 | 0.608 | 0.025 |
| grid | Fixed CLAHE (CL=2.0) | 10 | 1.626 | 0.931 | 0.609 | 0.019 |
| grid | Rule-based Adaptive CLAHE | 10 | 1.742 | 0.908 | 0.569 | 0.026 |
| grid | Proposed DT-QAT Student | 10 | 1.451 | 0.942 | 0.605 | 0.020 |

## Command Arguments
- `Namespace(mvtec_root=WindowsPath('project/datasets/mvtec_anomaly_detection_subset'), pairs_csv=None, categories=['carpet', 'grid'], output_root=WindowsPath('project/RL_CLAHE/analysis/masked_industrial_compare/out/mvtec_carpet_grid_20'), max_per_defect=0, max_items=0, resize='', skip_student=False, panel_count=8)`
