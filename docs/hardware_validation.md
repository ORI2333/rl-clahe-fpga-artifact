# Hardware Validation Notes

## Current Status

Existing routed reports were found for the MLP controller, but they should not
be used as final complete-system evidence yet:

- Total on-chip power: 1.061 W.
- Dynamic power: 0.897 W.
- Static power: 0.164 W.
- Power confidence: Low.
- 100 MHz routed timing WNS: -0.448 ns.

## Next Evidence Target

Before writing final power/energy claims, rerun implementation after timing or
clock-policy cleanup and record:

- FPGA part and board.
- Vivado version.
- Clock frequency and timing status.
- Activity assumptions or switching data.
- Utilization, timing, power, energy/frame, and energy/pixel.
- Whether numbers are controller-only or full CLAHE pipeline.

## Board-Level Smoke Test

After RTL/golden consistency:

- Build bitstream.
- Feed a fixed frame or video stream.
- Capture output frame or clip-limit trajectory.
- Compare with the software/RTL golden trajectory.

