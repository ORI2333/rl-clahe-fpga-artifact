# Frame-Level Clip-Limit Smoothing Diagnostic

## Scope

- Purpose: address the reviewer concern about one-frame update delay under rapid illumination changes.
- Implementation excerpt: `rtl/controller/rl_clip_limit_smoother.sv`.
- Integration in the working FPGA prototype: `actor5_ctrl_phys.sv` uses threshold-triggered smoothing on the final per-frame `CLIP_LIMIT`.
- Default policy: if `abs(target_clip - previous_smooth_clip) <= 256`, follow the target directly; otherwise move by at most `128` per frame.
- This is an RTL/Python diagnostic and an engineering improvement candidate. It is not board-level HIL evidence.

## Evidence Files

- `japan_lighting_30f_smoothing_expected.csv`
- `norway_fadein_30f_smoothing_expected.csv`

The raw videos and generated full pixel streams are not redistributed in this
public artifact. The CSVs are derived diagnostic outputs.

## Results

| Segment | Frames | Threshold | Step max | Triggered frames | Max delta | First smooth | Last smooth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Japan lighting-change | 30 | 256 | 128 | 2 | 452 | 772 | 638 |
| Norway fade-in | 30 | 256 | 128 | 5 | 836 | 772 | 295 |

## Interpretation

- Japan lighting-change mostly follows the target directly; smoothing is only triggered at the initial large drop from the default clip limit and the next large correction.
- Norway fade-in has a stronger early transition toward the low clip-limit clamp, so smoothing protects the first five frames and then follows the target directly.
- This supports a conservative response: the implemented frame-boundary update can be paired with threshold-triggered inter-frame damping for abrupt transitions, while normal gradual changes are not unnecessarily delayed.

## Verification

- `rl_stream_phys_compare.py` compiles with Python.
- Modified SystemVerilog files pass `xvlog -sv` syntax analysis:
  - `rtl/controller/rl_clip_limit_smoother.sv`
  - `rtl/rl/actor5_ctrl_phys.sv`
  - `rtl/rl_frontend/rl_stream_policy_phys_top.sv`
  - `rtl/isp_passthrough.sv`
