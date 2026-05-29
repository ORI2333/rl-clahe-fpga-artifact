`timescale 1ns/1ps
`include "defines.svh"

module layer4_dense(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         start_pulse,

    // L3H RAM 读口
    output logic [$clog2(`DIM_L3H)-1:0]  l3h_rd_addr,
    input  logic signed [15:0]           l3h_rd_data,

    output logic                         done_pulse,
    output logic signed [15:0]           y_out_q12
);
    localparam int NIN = `DIM_L3H;

    logic [$clog2(NIN)-1:0]  cnt;
    logic signed [15:0]      w_data;
    logic signed [15:0]      b_data;

    small_rom #(.HEX_PATH("F:/EngineeringWarehouse/ISP/RL/2.FPGA/MLP_V2.0/hex/l3_w.hex"), .DEPTH(NIN)) u_w(
        .clka(clk), .addra(cnt), .douta(w_data)
    );
    small_rom #(.HEX_PATH("F:/EngineeringWarehouse/ISP/RL/2.FPGA/MLP_V2.0/hex/l3_b.hex"), .DEPTH(1), .ADDR_WIDTH(1)) u_b(
        .clka(clk), .addra(1'b0), .douta(b_data)
    );

    logic mac_done;


    logic active, rd_vld;   // rd_vld=1 表示读口数据已对齐可用（补偿1拍延迟）
    // 推动读口
    assign l3h_rd_addr = cnt;
    // 计数与读使能
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cnt <= '0; active <= 1'b0; rd_vld <= 1'b0;
    end else begin
        if (start_pulse) begin
            // 第1拍：置 addr=0，等待读口数据出现在下一拍
            active <= 1'b1;
            rd_vld <= 1'b0;
            cnt    <= '0;
        end else if (active) begin
            if (!rd_vld) begin
                // 第2拍：数据对齐，此拍起可以启动 MAC
                rd_vld <= 1'b1;
            end else begin
                // 之后每拍递增地址，直至 NIN-1
                if (cnt != NIN-1) cnt <= cnt + 1'b1;
                // MAC 完成再清空 active（避免重复启动）
                else if (mac_done) active <= 1'b0;
            end
        end
    end
end
// 仅在“数据对齐且地址仍为0”的这一拍拉起 start_in（1拍）
wire start_for_mac = active && rd_vld && (cnt == '0);

neuron_mac #(.INPUT_COUNT(NIN), .ENABLE_RELU(1'b0)) u_mac (
    .clk(clk), .rst_n(rst_n),
    .start_in  (start_for_mac),
    .feature_in(l3h_rd_data),
    .weight_in (w_data),
    .bias_in   (b_data),
    .done_out  (mac_done),
    .result_out(y_out_q12)
);

    assign done_pulse = mac_done;
endmodule
