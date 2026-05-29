# TECS CLAHE Review Artifact

This repository is a private working artifact for the TECS major revision of
"Stall-Free Software--Hardware Co-Design for Perception-Driven CLAHE".

The goal is to keep a clean, review-ready package that can later be mirrored to
an anonymized review repository or submitted as supplementary material.

## Artifact Scope

Included:

- Scripts for image-quality evaluation, tail-robustness summaries, and ablation summaries.
- Student-distillation and fixed-point export/check scripts.
- Representative fixed-point RTL for the adaptive controller.
- Fixed-point hex artifacts used by the FPGA controller.
- CSV outputs used to reproduce key tables in the revised paper.
- Notes for reviewers and maintainers.

Not included yet:

- Full training datasets.
- Uncleaned Vivado project caches, run directories, and generated IP folders.
- Large model checkpoints that still need license/path review.
- Board-specific bitstreams and internal lab paths.

## Current Review Strategy

During review, this artifact should be shared as an anonymized, review-only
package. After acceptance, a cleaned archival release can be published with a
standard public license and DOI.

## Quick Start

The package is being assembled. The stable entry points are:

```bash
python software/experiment/run_ablation_on_testset_V4.py
python software/distillation/verify_student_5step_hw_like.py
python software/distillation/export_student_q12_multihead_V2.py
```

Some scripts require optional model checkpoints or datasets that are not yet
bundled. See `docs/reproducibility_plan.md` for the completion checklist.

## Mapping to Paper Revision

- R1-2: Fixed CLAHE sensitivity, scene-wise behavior, and tail robustness.
- R1-3 / R2-2: Hardware resource, timing, power, and energy evidence.
- R3: RTL/golden consistency, quantization sensitivity, code availability, and limitations.

