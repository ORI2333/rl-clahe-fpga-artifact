# Industrial Texture / Surface-Defect Diagnostic

## Dataset
- Local sample root: not redistributed in this artifact; the local sample was reconstructed from the source listed below.
- Source used for this local diagnostic: GitHub mirror of NEU-DET steel surface defect images, `siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection`, `IMAGES/`.
- Intended interpretation: supplemental diagnostic for highly textured industrial scenes; do not redistribute source images in the review artifact until license/source terms are finalized.
- Images analyzed: 72
- Resize: `native`

## Evidence Boundary
- Industrial surface-defect images are not natural-scene photographs, so BRISQUE/NIQE are kept only as secondary natural-scene-statistics diagnostics when enabled.
- No pixel-level masks or detector are used here, so the metrics below are visibility/amplification diagnostics rather than detection accuracy.
- MAE/SSIM are relative to the input image and only measure structural deviation, not absolute enhancement quality.
- RMS contrast, robust Michelson contrast, Tenengrad energy, edge density, local standard deviation, Laplacian variance, and high-frequency ratio are used as industrial-texture diagnostics.
- Gains are computed relative to the input; high gains indicate stronger texture/edge/high-frequency amplification, which can improve visibility but may also amplify noise/artifacts.

## Overall Industrial-Diagnostic Summary
| Method | CL mean | RMS gain | Michelson gain | Tenengrad gain | Edge density | Edge delta (pp) | Local-std gain | Lap-var gain | HF gain | MAE | SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Input | nan | 1.000 | 1.000 | 1.000 | 0.157 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| Global HE | nan | 3.344 | 3.501 | 12.262 | 0.321 | 16.433 | 3.346 | 16.699 | 10.418 | 46.869 | 0.578 |
| Fixed CLAHE (CL=2.0) | 2.000 | 1.253 | 1.277 | 3.938 | 0.250 | 9.296 | 2.040 | 4.350 | 3.954 | 13.962 | 0.869 |
| Fixed CLAHE (CL=4.0) | 4.000 | 1.647 | 1.746 | 9.568 | 0.332 | 17.492 | 3.186 | 11.397 | 10.012 | 26.185 | 0.692 |
| Rule-based Adaptive CLAHE | 8.790 | 2.275 | 2.495 | 26.354 | 0.379 | 22.175 | 5.189 | 36.519 | 26.764 | 40.121 | 0.478 |
| Adaptive Gamma Correction | 0.975 | 1.002 | 0.980 | 1.003 | 0.162 | 0.486 | 1.003 | 1.004 | 0.970 | 5.118 | 0.993 |
| MSR Retinex | nan | 1.304 | 1.224 | 4.178 | 0.228 | 7.090 | 1.789 | 4.073 | 3.236 | 38.432 | 0.842 |
| Proposed DT-QAT Student | 3.532 | 1.551 | 1.669 | 8.197 | 0.311 | 15.365 | 2.835 | 9.865 | 9.267 | 22.375 | 0.739 |

## Secondary Natural-Scene Diagnostics
| Method | BRISQUE | NIQE | Mean | Std | Entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Input | nan | nan | 134.190 | 27.982 | 6.482 |
| Global HE | nan | nan | 129.428 | 73.482 | 6.379 |
| Fixed CLAHE (CL=2.0) | nan | nan | 135.379 | 33.332 | 6.850 |
| Fixed CLAHE (CL=4.0) | nan | nan | 133.057 | 42.105 | 7.225 |
| Rule-based Adaptive CLAHE | nan | nan | 131.658 | 54.373 | 7.644 |
| Adaptive Gamma Correction | nan | nan | 137.660 | 28.064 | 6.466 |
| MSR Retinex | nan | nan | 156.461 | 30.868 | 6.733 |
| Proposed DT-QAT Student | nan | nan | 131.167 | 39.901 | 7.124 |

## By-Class Summary
| Class | Method | n | CL mean | RMS gain | Tenengrad gain | Edge density | Edge delta (pp) | Local-std gain | Lap-var gain | MAE | SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| crazing | Input | 12 | nan | 1.000 | 1.000 | 0.373 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| crazing | Global HE | 12 | nan | 2.485 | 6.320 | 0.376 | 0.277 | 2.482 | 6.613 | 45.183 | 0.666 |
| crazing | Fixed CLAHE (CL=2.0) | 12 | 2.000 | 1.469 | 3.724 | 0.391 | 1.727 | 1.938 | 3.808 | 22.747 | 0.817 |
| crazing | Fixed CLAHE (CL=4.0) | 12 | 4.000 | 2.093 | 8.165 | 0.388 | 1.518 | 2.874 | 8.651 | 40.952 | 0.598 |
| crazing | Rule-based Adaptive CLAHE | 12 | 8.055 | 2.397 | 11.144 | 0.386 | 1.275 | 3.347 | 12.141 | 50.082 | 0.531 |
| crazing | Adaptive Gamma Correction | 12 | 1.000 | 1.000 | 1.000 | 0.373 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| crazing | MSR Retinex | 12 | nan | 0.946 | 1.271 | 0.378 | 0.477 | 1.109 | 1.237 | 19.130 | 0.947 |
| crazing | Proposed DT-QAT Student | 12 | 3.386 | 1.953 | 6.357 | 0.385 | 1.143 | 2.440 | 6.787 | 34.216 | 0.702 |
| inclusion | Input | 12 | nan | 1.000 | 1.000 | 0.004 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| inclusion | Global HE | 12 | nan | 5.977 | 35.527 | 0.319 | 31.440 | 6.059 | 49.529 | 56.508 | 0.321 |
| inclusion | Fixed CLAHE (CL=2.0) | 12 | 2.000 | 1.270 | 5.061 | 0.033 | 2.854 | 2.342 | 5.973 | 8.727 | 0.830 |
| inclusion | Fixed CLAHE (CL=4.0) | 12 | 4.000 | 1.771 | 13.145 | 0.257 | 25.303 | 3.888 | 17.739 | 17.571 | 0.704 |
| inclusion | Rule-based Adaptive CLAHE | 12 | 10.598 | 2.866 | 58.034 | 0.376 | 37.139 | 8.247 | 89.636 | 30.939 | 0.384 |
| inclusion | Adaptive Gamma Correction | 12 | 0.900 | 0.985 | 0.975 | 0.004 | -0.014 | 0.986 | 0.974 | 10.821 | 0.985 |
| inclusion | MSR Retinex | 12 | nan | 2.101 | 12.401 | 0.061 | 5.717 | 3.051 | 11.251 | 82.049 | 0.651 |
| inclusion | Proposed DT-QAT Student | 12 | 3.954 | 1.799 | 13.887 | 0.272 | 26.780 | 3.972 | 18.628 | 17.958 | 0.693 |
| patches | Input | 12 | nan | 1.000 | 1.000 | 0.314 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| patches | Global HE | 12 | nan | 1.392 | 2.246 | 0.336 | 2.140 | 1.449 | 2.324 | 29.630 | 0.906 |
| patches | Fixed CLAHE (CL=2.0) | 12 | 2.000 | 1.019 | 3.114 | 0.379 | 6.465 | 1.742 | 3.325 | 19.786 | 0.905 |
| patches | Fixed CLAHE (CL=4.0) | 12 | 4.000 | 1.144 | 6.111 | 0.383 | 6.894 | 2.399 | 6.808 | 34.472 | 0.746 |
| patches | Rule-based Adaptive CLAHE | 12 | 6.771 | 1.209 | 8.245 | 0.383 | 6.843 | 2.759 | 9.397 | 42.778 | 0.637 |
| patches | Adaptive Gamma Correction | 12 | 0.925 | 0.989 | 0.961 | 0.322 | 0.761 | 0.991 | 0.956 | 6.935 | 0.990 |
| patches | MSR Retinex | 12 | nan | 0.648 | 0.602 | 0.250 | -6.479 | 0.754 | 0.581 | 59.714 | 0.767 |
| patches | Proposed DT-QAT Student | 12 | 4.167 | 1.114 | 4.483 | 0.362 | 4.749 | 1.982 | 5.082 | 26.314 | 0.817 |
| pitted_surface | Input | 12 | nan | 1.000 | 1.000 | 0.082 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| pitted_surface | Global HE | 12 | nan | 2.449 | 6.249 | 0.247 | 16.452 | 2.429 | 6.411 | 47.571 | 0.678 |
| pitted_surface | Fixed CLAHE (CL=2.0) | 12 | 2.000 | 1.065 | 5.014 | 0.268 | 18.587 | 2.210 | 5.127 | 9.696 | 0.928 |
| pitted_surface | Fixed CLAHE (CL=4.0) | 12 | 4.000 | 1.198 | 14.410 | 0.330 | 24.726 | 3.742 | 14.940 | 23.240 | 0.722 |
| pitted_surface | Rule-based Adaptive CLAHE | 12 | 8.075 | 1.639 | 38.452 | 0.376 | 29.320 | 6.143 | 41.415 | 39.686 | 0.440 |
| pitted_surface | Adaptive Gamma Correction | 12 | 1.075 | 1.053 | 1.127 | 0.104 | 2.167 | 1.055 | 1.131 | 7.575 | 0.989 |
| pitted_surface | MSR Retinex | 12 | nan | 1.496 | 5.726 | 0.295 | 21.257 | 2.430 | 5.877 | 18.343 | 0.894 |
| pitted_surface | Proposed DT-QAT Student | 12 | 3.601 | 1.298 | 13.973 | 0.261 | 17.877 | 3.365 | 14.620 | 23.470 | 0.722 |
| rolled-in_scale | Input | 12 | nan | 1.000 | 1.000 | 0.149 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| rolled-in_scale | Global HE | 12 | nan | 3.917 | 15.539 | 0.368 | 21.897 | 3.865 | 16.386 | 49.238 | 0.476 |
| rolled-in_scale | Fixed CLAHE (CL=2.0) | 12 | 2.000 | 1.582 | 4.774 | 0.373 | 22.329 | 2.202 | 4.963 | 15.993 | 0.803 |
| rolled-in_scale | Fixed CLAHE (CL=4.0) | 12 | 4.000 | 2.331 | 12.256 | 0.383 | 23.360 | 3.535 | 13.145 | 25.986 | 0.588 |
| rolled-in_scale | Rule-based Adaptive CLAHE | 12 | 9.191 | 3.524 | 30.801 | 0.382 | 23.231 | 5.609 | 35.030 | 47.803 | 0.367 |
| rolled-in_scale | Adaptive Gamma Correction | 12 | 1.000 | 1.000 | 1.000 | 0.149 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| rolled-in_scale | MSR Retinex | 12 | nan | 1.385 | 3.006 | 0.349 | 19.953 | 1.704 | 2.924 | 20.012 | 0.900 |
| rolled-in_scale | Proposed DT-QAT Student | 12 | 2.640 | 1.829 | 7.372 | 0.380 | 23.015 | 2.731 | 7.792 | 19.523 | 0.675 |
| scratches | Input | 12 | nan | 1.000 | 1.000 | 0.019 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| scratches | Global HE | 12 | nan | 3.844 | 7.692 | 0.282 | 26.392 | 3.791 | 18.930 | 53.086 | 0.419 |
| scratches | Fixed CLAHE (CL=2.0) | 12 | 2.000 | 1.115 | 1.939 | 0.057 | 3.815 | 1.806 | 2.902 | 6.826 | 0.930 |
| scratches | Fixed CLAHE (CL=4.0) | 12 | 4.000 | 1.343 | 3.321 | 0.250 | 23.148 | 2.676 | 7.100 | 14.888 | 0.792 |
| scratches | Rule-based Adaptive CLAHE | 12 | 10.050 | 2.014 | 11.450 | 0.371 | 35.243 | 5.030 | 31.497 | 29.439 | 0.510 |
| scratches | Adaptive Gamma Correction | 12 | 0.950 | 0.985 | 0.953 | 0.019 | 0.000 | 0.985 | 0.966 | 5.376 | 0.993 |
| scratches | MSR Retinex | 12 | nan | 1.248 | 2.063 | 0.035 | 1.619 | 1.686 | 2.567 | 31.344 | 0.893 |
| scratches | Proposed DT-QAT Student | 12 | 3.442 | 1.311 | 3.106 | 0.205 | 18.629 | 2.519 | 6.281 | 12.771 | 0.826 |

## Outputs
- Detail CSV: `results/industrial_texture_diagnostic/method_detail.csv`
- Overall summary CSV: `results/industrial_texture_diagnostic/method_summary_overall.csv`
- By-class summary CSV: `results/industrial_texture_diagnostic/method_summary_by_class.csv`
- Visual panel: `results/industrial_texture_diagnostic/industrial_representative_panel.png`
- `frames/`: per-method grayscale output images.
