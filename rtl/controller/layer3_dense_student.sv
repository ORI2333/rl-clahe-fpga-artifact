`timescale 1ns/1ps
`include "defines.svh"

// 学生版输出层：DIM_L2(64) -> 1
module layer3_dense_student(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        start_pulse,

    // L2 RAM 读口（你可以用 dp_ram B 口接进来）
    output logic [$clog2(`DIM_L2)-1:0]  l2_ram_addr,
    input  logic signed [15:0]          l2_ram_data,

    output logic                        done_pulse,
    output logic signed [15:0]          y_out_q12
);
    localparam int NIN = `DIM_L2;

    logic [$clog2(NIN)-1:0] cnt;
    logic signed [15:0]     w_data;
    logic signed [15:0]     b_data;

    // 换成自己的路径
    small_rom #(
        .HEX_PATH("hex/l3_w.hex"),
        .DEPTH   (NIN)
    ) u_w (
        .clka  (clk),
        .addra (cnt),
        .douta (w_data)
    );

    small_rom #(
        .HEX_PATH  ("hex/l3_b.hex"),
        .DEPTH     (1),
        .ADDR_WIDTH(1)
    ) u_b (
        .clka  (clk),
        .addra (1'b0),
        .douta (b_data)
    );

    logic mac_done;
    logic active, rd_vld;

    assign l2_ram_addr = cnt;

    // 读口对齐 + 计数
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt    <= '0;
            active <= 1'b0;
            rd_vld <= 1'b0;
        end else begin
            if (start_pulse) begin
                active <= 1'b1;
                rd_vld <= 1'b0;
                cnt    <= '0;
            end else if (active) begin
                if (!rd_vld) begin
                    rd_vld <= 1'b1;
                end else begin
                    if (cnt != NIN-1)
                        cnt <= cnt + 1'b1;
                    else if (mac_done)
                        active <= 1'b0;
                end
            end
        end
    end

    wire start_for_mac = active && rd_vld && (cnt == '0);

    neuron_mac #(
        .INPUT_COUNT (NIN),
        .ENABLE_RELU (1'b0)
    ) u_mac (
        .clk        (clk),
        .rst_n      (rst_n),
        .start_in   (start_for_mac),
        .feature_in (l2_ram_data),
        .weight_in  (w_data),
        .bias_in    (b_data),
        .done_out   (mac_done),
        .result_out (y_out_q12)
    );

    assign done_pulse = mac_done;
endmodule
