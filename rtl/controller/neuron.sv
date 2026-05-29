`timescale 1ns/1ps
`include "defines.svh"

// 通用 INPUT_COUNT 的 neuron：Q4.12 → Q8.24 累加 → 加 bias → 回 Q4.12
module neuron #(
    parameter int INPUT_COUNT = `DIM_IN,
    parameter bit ENABLE_RELU = 1'b1
)(
    input  logic                           clk,
    input  logic                           rst_n,
    input  logic                           data_valid_in,
    input  logic signed [15:0]             feature_in   [INPUT_COUNT],
    input  logic signed [15:0]             weights_in   [INPUT_COUNT],
    input  logic signed [15:0]             bias_in,
    output logic                           data_valid_out,
    output logic signed [15:0]             result_out
);
    localparam int FRAC = `Q_FRAC_BITS;   // 一般是 12

    // Q4.12 * Q4.12 = Q8.24
    function automatic logic signed [31:0] mul_q12_to_q24(
        input logic signed [15:0] a_q12,
        input logic signed [15:0] b_q12
    );
        return a_q12 * b_q12;
    endfunction

    // 饱和到 16bit Q4.12
    function automatic logic signed [15:0] sat16_q12(input logic signed [31:0] x);
        if (x >  32'sd32767)  return 16'sh7FFF;
        if (x < -32'sd32768)  return 16'sh8000;
        return x[15:0];
    endfunction

    // pipeline 标志
    logic               v1, v2, v3;
    // 逐元素乘积 Q8.24（寄存器）
    logic signed [31:0] p [INPUT_COUNT];
    // 累加和：寄存器 + 组合
    logic signed [47:0] sum_q24;        // 寄存器：上一拍 Σp
    logic signed [47:0] sum_q24_next;   // 组合：这一拍 Σp
    logic signed [47:0] sum_bias_q24;   // 加完 bias
    // Q4.12 对齐结果
    logic signed [31:0] q_shift;
    logic signed [15:0] q_q12;
    logic signed [15:0] bias_latch;

    integer i;

    // ---------------- Stage1: 乘法 + bias latch ----------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0;
            for (i = 0; i < INPUT_COUNT; i++) p[i] <= '0;
            bias_latch <= '0;
        end else begin
            v1 <= data_valid_in;
            if (data_valid_in) begin
                for (i = 0; i < INPUT_COUNT; i++) begin
                    p[i] <= mul_q12_to_q24(feature_in[i], weights_in[i]); // Q8.24
                end
                bias_latch <= bias_in;
            end
        end
    end

    // ---------------- 组合：Σp[i] → sum_q24_next ----------------
    always_comb begin
        sum_q24_next = '0;
        for (int k = 0; k < INPUT_COUNT; k++) begin
            // 32bit Q8.24 → 48bit Q8.24
            sum_q24_next += {{16{p[k][31]}}, p[k]};
        end
    end

    // ---------------- Stage2: 打拍 Σp ----------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v2      <= 1'b0;
            sum_q24 <= '0;
        end else begin
            v2      <= v1;
            sum_q24 <= sum_q24_next;
        end
    end

// ---------------- Stage3: 加 bias，对齐到 Q8.24 ----------------
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        v3           <= 1'b0;
        sum_bias_q24 <= '0;
    end else begin
        v3           <= v2;
        sum_bias_q24 <= sum_q24
                      + ({{(48-16){bias_latch[15]}}, bias_latch} <<< FRAC);
    end
end

// ---------- 组合：从 sum_bias_q24 计算本拍的 q_shift / q_q12 ----------
logic signed [31:0] q_shift_next;
logic signed [15:0] q_q12_next;

always_comb begin
    q_shift_next = sum_bias_q24 >>> FRAC;      // Q8.24 → Q4.12
    q_q12_next   = sat16_q12(q_shift_next);    // 饱和到 16bit
end

// ---------------- Stage4: 打拍 + ReLU + data_valid ----------------
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        q_shift        <= '0;
        q_q12          <= '0;
        data_valid_out <= 1'b0;
        result_out     <= '0;
    end else begin
        // 打拍保存本拍的计算结果（便于调试、级联）
        q_shift <= q_shift_next;
        q_q12   <= q_q12_next;

        // 有效标志和 sum_bias_q24 同拍
        data_valid_out <= v3;

        // ReLU 用“本拍的” q_q12_next
        if (ENABLE_RELU && q_q12_next[15])
            result_out <= 16'sd0;
        else
            result_out <= q_q12_next;
    end
end


endmodule
