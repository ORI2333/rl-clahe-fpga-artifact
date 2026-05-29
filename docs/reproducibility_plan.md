# Reproducibility Plan

## Goal

Build a review artifact that lets reviewers reproduce the revised paper's key
claims without exposing uncleaned history, personal paths, private datasets, or
unfinished Vivado state.

## Must Reproduce Before Submission

- Fixed CLAHE clip-limit sweep.
- Scene-wise summary and tail-robustness metrics.
- Component ablation summary.
- Student fixed-point export to Q4.12.
- RTL/golden consistency for representative observations.
- Vivado utilization/timing/power reports after timing or clock-policy cleanup.

## Current Included Files

- `software/experiment`: evaluation and table-generation scripts.
- `software/distillation`: distillation, export, and verification scripts.
- `rtl/controller`: SystemVerilog adaptive-controller implementation.
- `rtl/preprocessing`: hardware-observable feature modules.
- `results`: selected CSV outputs and hardware-report placeholders.

## Files Still Needing Review

- Model checkpoints in `distill_out` and teacher checkpoints.
- Dataset samples that can be redistributed.
- Board-level HIL logs and bitstream policy.
- Final public license selection after acceptance.

## Anonymization Checklist

- No author names, university names, personal GitHub accounts, or local Windows paths.
- No commit history copied from development repositories.
- No links to legacy preliminary repositories.
- No personal productivity-tool notes, API keys, or machine-specific configuration.
- Run a text scan for local paths and secrets before mirroring.
