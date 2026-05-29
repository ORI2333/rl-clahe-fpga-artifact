`timescale 1ns/1ps
`include "defines.svh"

// 学生版 MLP： DIM_IN(5) -> DIM_L1(128) -> DIM_L2(64) -> 1
module mlp_student_top_2h(
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     start_pulse,
    input  logic signed [15:0]       obs_q12 [`DIM_IN],

    output logic                     y_valid,
    output logic signed [15:0]       y_q12
);
    // ---------- L1 RAM ----------
    logic                       l1_we;
    logic [$clog2(`DIM_L1)-1:0] l1_waddr;
    logic signed [15:0]         l1_wdata;
    logic                       l1_done;

    logic [$clog2(`DIM_L1)-1:0] l2_l1_rd_addr;
    logic signed [15:0]         l2_l1_rd_data;

        // ---------- L2 RAM ----------
    // L2 层把 64 维输出写进 RAM，供输出层逐个读
    logic                         l2_we;
    logic [$clog2(`DIM_L2)-1:0]   l2_waddr;
    logic signed [15:0]           l2_wdata;

    // 提供给输出层的读口
    logic [$clog2(`DIM_L2)-1:0]   l3_l2_rd_addr;
    logic signed [15:0]           l3_l2_rd_data;

    dp_ram #(.DEPTH(`DIM_L1)) u_l1_ram (
        .clka   (clk),
        .wea    (l1_we),
        .addra  (l1_waddr),
        .dina   (l1_wdata),
        .clkb   (clk),
        .addrb  (l2_l1_rd_addr),
        .doutb  (l2_l1_rd_data)
    );

    neuron_layer_l1 u_layer1 (
        .clk               (clk),
        .rst_n             (rst_n),
        .start_pulse       (start_pulse),
        .feature_in        (obs_q12),
        .ram_we            (l1_we),
        .ram_waddr         (l1_waddr),
        .ram_wdata         (l1_wdata),
        .layer_done_pulse  (l1_done)
    );



    dp_ram #(
        .DEPTH      (`DIM_L2)
    ) u_l2_ram (
        .clka       (clk),
        .wea        (l2_we),
        .addra      (l2_waddr),
        .dina       (l2_wdata),

        .clkb       (clk),
        .addrb      (l3_l2_rd_addr),
        .doutb      (l3_l2_rd_data)
    );

    // 学生版 L2：128 -> 64，单 batch
    neuron_layer_l2_student u_layer2 (
        .clk               (clk),
        .rst_n             (rst_n),
        .start_pulse       (l1_done),
        .l1_rd_addr        (l2_l1_rd_addr),
        .l1_rd_data        (l2_l1_rd_data),
        .l2_we             (l2_we),
        .l2_waddr          (l2_waddr),
        .l2_wdata          (l2_wdata),
        .layer_done_pulse  (l2_done)
    );
    // ---------- 输出层：64 -> 1 ----------
    layer3_dense_student u_layer3 (
        .clk         (clk),
        .rst_n       (rst_n),
        .start_pulse (l2_done),

        .l2_ram_addr (l3_l2_rd_addr),
        .l2_ram_data (l3_l2_rd_data),

        .y_out_q12   (y_q12),
        .done_pulse  (y_valid)
    );

endmodule
