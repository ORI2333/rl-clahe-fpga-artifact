`timescale 1ns/1ps
`include "defines.svh"

// 7 -> 256 -> 256 -> 128 -> 1
module mlp_actor_top_3h(
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     start_pulse,
    input  logic signed [15:0]       obs7_q12 [`DIM_IN],

    output logic                     y_valid,
    output logic signed [15:0]       y_q12
);
    // L1 RAM
    logic                       l1_we;
    logic [$clog2(`DIM_L1)-1:0] l1_waddr;
    logic signed [15:0]         l1_wdata;
    logic                       l1_done;

    // L2 ? L1 / ? L2
    logic [$clog2(`DIM_L1)-1:0] l2_l1_rd_addr;
    logic signed [15:0]         l2_l1_rd_data;
    logic                       l2_we;
    logic [$clog2(`DIM_L2)-1:0] l2_waddr;
    logic signed [15:0]         l2_wdata;
    logic                       l2_done;

    // L3H ? L2 / ? L3H
    logic [$clog2(`DIM_L2)-1:0]  l3_l2_rd_addr;
    logic signed [15:0]          l3_l2_rd_data;
    logic                        l3h_we;
    logic [$clog2(`DIM_L3H)-1:0] l3h_waddr;
    logic signed [15:0]          l3h_wdata;
    logic                        l3h_done;

    // L4 ? L3H
    logic [$clog2(`DIM_L3H)-1:0] l4_l3h_rd_addr;
    logic signed [15:0]          l4_l3h_rd_data;

    // ---------- L1 RAM ----------
    dp_ram #(.DEPTH(`DIM_L1)) u_l1_ram (
        .clka   (clk),
        .wea    (l1_we),
        .addra  (l1_waddr),
        .dina   (l1_wdata),
        .clkb   (clk),
        .addrb  (l2_l1_rd_addr),
        .doutb  (l2_l1_rd_data)
    );

    // ---------- L1 ? ----------
    neuron_layer_l1 u_layer1 (
        .clk               (clk),
        .rst_n             (rst_n),
        .start_pulse       (start_pulse),
        .feature_in        (obs7_q12),
        .ram_we            (l1_we),
        .ram_waddr         (l1_waddr),
        .ram_wdata         (l1_wdata),
        .layer_done_pulse  (l1_done)
    );

    // ---------- L2 RAM ----------
    dp_ram #(.DEPTH(`DIM_L2)) u_l2_ram (
        .clka   (clk),
        .wea    (l2_we),
        .addra  (l2_waddr),
        .dina   (l2_wdata),
        .clkb   (clk),
        .addrb  (l3_l2_rd_addr),
        .doutb  (l3_l2_rd_data)
    );

    // ---------- L2 ? ----------
    neuron_layer_l2 u_layer2 (
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

    // ---------- L3H RAM ----------
    dp_ram #(.DEPTH(`DIM_L3H)) u_l3h_ram (
        .clka   (clk),
        .wea    (l3h_we),
        .addra  (l3h_waddr),
        .dina   (l3h_wdata),
        .clkb   (clk),
        .addrb  (l4_l3h_rd_addr),
        .doutb  (l4_l3h_rd_data)
    );

    // ---------- L3H ? ----------
    neuron_layer_l3h u_layer3h (
        .clk               (clk),
        .rst_n             (rst_n),
        .start_pulse       (l2_done),
        .l2_rd_addr        (l3_l2_rd_addr),
        .l2_rd_data        (l3_l2_rd_data),
        .l3h_we            (l3h_we),
        .l3h_waddr         (l3h_waddr),
        .l3h_wdata         (l3h_wdata),
        .layer_done_pulse  (l3h_done)
    );

    // ---------- L4 ? ----------
    layer4_dense u_layer4 (
        .clk           (clk),
        .rst_n         (rst_n),
        .start_pulse   (l3h_done),
        .l3h_rd_addr   (l4_l3h_rd_addr),
        .l3h_rd_data   (l4_l3h_rd_data),
        .done_pulse    (y_valid),
        .y_out_q12     (y_q12)
    );
endmodule
