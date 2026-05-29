`timescale 1ns/1ps
`include "defines.svh"

// ============================================================================
// 演示顶层：提供一个固定 obs7_in（可在 TB 中驱动），观察 5 步后的 final_cl_q12
// ============================================================================
module top_5step_demo(
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start_pulse,
    input  logic signed [15:0] obs_in [`DIM_IN],
    output logic               final_valid,
    output logic signed [15:0] final_cl_q12
);
    actor5_ctrl u_ctrl(
        .clk                               (clk                            ),
        .rst_n                             (rst_n                          ),
        .start_pulse                       (start_pulse                    ),
        .obs_in                           (obs_in                        ),
        .final_valid                       (final_valid                    ),
        .final_cl_q12                      (final_cl_q12                   ) 
    );
endmodule
