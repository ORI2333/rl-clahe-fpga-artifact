`timescale 1ns / 1ns

module rl_clip_limit_smoother #(
    parameter integer ENABLE_SLEW_LIMIT          = 1,
    parameter integer CLIP_LIMIT_STEP_MAX        = 128,
    parameter integer CLIP_LIMIT_SLEW_THRESHOLD  = 256
) (
    input  wire [31:0] target_clip,
    input  wire [31:0] previous_clip,
    output reg  [31:0] smooth_clip,
    output reg         smoothing_triggered,
    output reg  [31:0] delta_abs
);
    localparam [31:0] CLIP_LIMIT_STEP_MAX_U = CLIP_LIMIT_STEP_MAX;
    localparam [31:0] CLIP_LIMIT_SLEW_THRESHOLD_U = CLIP_LIMIT_SLEW_THRESHOLD;

    always @* begin
        smooth_clip = target_clip;
        smoothing_triggered = 1'b0;

        if (target_clip >= previous_clip) begin
            delta_abs = target_clip - previous_clip;
            if ((ENABLE_SLEW_LIMIT != 0) && (delta_abs > CLIP_LIMIT_SLEW_THRESHOLD_U)) begin
                smoothing_triggered = 1'b1;
                if (delta_abs > CLIP_LIMIT_STEP_MAX_U) begin
                    smooth_clip = previous_clip + CLIP_LIMIT_STEP_MAX_U;
                end else begin
                    smooth_clip = target_clip;
                end
            end
        end else begin
            delta_abs = previous_clip - target_clip;
            if ((ENABLE_SLEW_LIMIT != 0) && (delta_abs > CLIP_LIMIT_SLEW_THRESHOLD_U)) begin
                smoothing_triggered = 1'b1;
                if (delta_abs > CLIP_LIMIT_STEP_MAX_U) begin
                    smooth_clip = previous_clip - CLIP_LIMIT_STEP_MAX_U;
                end else begin
                    smooth_clip = target_clip;
                end
            end
        end
    end
endmodule
