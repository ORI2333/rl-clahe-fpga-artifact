`timescale 1ns/1ps
`include "defines.svh"

// L3H: 256 -> 128
module neuron_layer_l3h(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         start_pulse,

    // L2 RAM 读口
    output logic [$clog2(`DIM_L2)-1:0]   l2_rd_addr,
    input  logic signed [15:0]           l2_rd_data,

    // L3H RAM 写口
    output logic                         l3h_we,
    output logic [$clog2(`DIM_L3H)-1:0]  l3h_waddr,
    output logic signed [15:0]           l3h_wdata,

    output logic                         layer_done_pulse
);
    localparam int P      = `P_WIDTH;                 // 64
    localparam int BATCHN = (`DIM_L3H + P - 1) / P;   // 2
    localparam int MACN   = `DIM_L2;                  // 256

    typedef enum logic [1:0] {S_IDLE,S_COMP,S_WAIT,S_WRITE} st_t;
    st_t st_q, st_d;

    logic [$clog2(BATCHN)-1:0] batch_q, batch_d;
    logic [$clog2(MACN)-1:0]   mac_q,   mac_d;
    logic [$clog2(P)-1:0]      wr_q,    wr_d;

    logic [P*16-1:0] w_bus [BATCHN];
    logic [P*16-1:0] b_bus [BATCHN];

    rom_wide_hex #(.HEX_PATH("F:/EngineeringWarehouse/ISP/RL/2.FPGA/MLP_V2.0/hex/l3h_w_b0.hex"), .DEPTH(`DIM_L2), .WIDTH(P*16)) u_w0(.clka(clk), .addra(mac_q), .douta(w_bus[0]));
    rom_wide_hex #(.HEX_PATH("F:/EngineeringWarehouse/ISP/RL/2.FPGA/MLP_V2.0/hex/l3h_w_b1.hex"), .DEPTH(`DIM_L2), .WIDTH(P*16)) u_w1(.clka(clk), .addra(mac_q), .douta(w_bus[1]));

    rom_wide_hex #(.HEX_PATH("F:/EngineeringWarehouse/ISP/RL/2.FPGA/MLP_V2.0/hex/l3h_b_b0.hex"), .DEPTH(P), .WIDTH(P*16), .ADDR_WIDTH($clog2(P))) u_b0(.clka(clk), .addra(wr_q), .douta(b_bus[0]));
    rom_wide_hex #(.HEX_PATH("F:/EngineeringWarehouse/ISP/RL/2.FPGA/MLP_V2.0/hex/l3h_b_b1.hex"), .DEPTH(P), .WIDTH(P*16), .ADDR_WIDTH($clog2(P))) u_b1(.clka(clk), .addra(wr_q), .douta(b_bus[1]));

    wire [P*16-1:0] w_sel = w_bus[batch_q];
    wire [P*16-1:0] b_sel = b_bus[batch_q];

    logic [P-1:0]        mac_done_vec;
    logic signed [15:0]  mac_out_vec [P];

    // 读数据有效标志（补偿1拍延时）
    logic rd_vld_q, rd_vld_d;
    
    // 对齐后的启动脉冲（仅 1 拍）
    wire start_for_mac = (st_q == S_COMP) && rd_vld_q && (mac_q == '0);

    genvar i;
    generate
        for (i=0;i<P;i++) begin: G_MAC
            wire signed [15:0] w_i = w_sel[(i+1)*16-1 -: 16];
            wire signed [15:0] b_i = b_sel[(i+1)*16-1 -: 16];
            neuron_mac #(.INPUT_COUNT(`DIM_L2), .ENABLE_RELU(1'b1)) u_mac (
                .clk       (clk),
                .rst_n     (rst_n),
                .start_in  (start_for_mac),        // 修改：使用对齐脉冲
                .feature_in(l2_rd_data),
                .weight_in (w_i),
                .bias_in   (b_i),
                .done_out  (mac_done_vec[i]),
                .result_out(mac_out_vec[i])
            );
        end
    endgenerate

    assign l2_rd_addr = mac_q;

    always_comb begin
        st_d     = st_q;
        batch_d  = batch_q;
        mac_d    = mac_q;
        wr_d     = wr_q;
        rd_vld_d = rd_vld_q;

        case (st_q)
            S_IDLE: begin
                if (start_pulse) begin
                    st_d     = S_COMP;
                    batch_d  = '0;
                    mac_d    = '0;
                    wr_d     = '0;
                    rd_vld_d = 1'b0;  // 先清 0，下一拍对齐
                end
            end

            S_COMP: begin
                if (!rd_vld_q) begin
                    // 首拍：只做对齐，不发 MAC start，不递增地址
                    rd_vld_d = 1'b1;
                    mac_d    = '0;
                end else begin
                    // 对齐完成：此拍 start_for_mac=1，从 0 开始喂数，每拍 +1
                    if (mac_q == `DIM_L2-1) begin
                        st_d = S_WAIT;  // 发送完 0..255 后进等待
                    end
                    mac_d = mac_q + 1'b1;
                end
            end

            S_WAIT: begin
                if (&mac_done_vec) begin
                    st_d = S_WRITE;
                    wr_d = '0;
                end
            end

            S_WRITE: begin
                if (wr_q == P-1) begin
                    if (batch_q == BATCHN-1) begin
                        st_d = S_IDLE;
                    end else begin
                        st_d     = S_COMP;
                        batch_d  = batch_q + 1'b1;
                        mac_d    = '0;
                        rd_vld_d = 1'b0; // 下一批重新对齐
                    end
                    wr_d = '0;
                end else begin
                    wr_d = wr_q + 1'b1;
                end
            end

            default: st_d = S_IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st_q     <= S_IDLE;
            batch_q  <= '0;
            mac_q    <= '0;
            wr_q     <= '0;
            rd_vld_q <= 1'b0;
        end else begin
            st_q     <= st_d;
            batch_q  <= batch_d;
            mac_q    <= mac_d;
            wr_q     <= wr_d;
            rd_vld_q <= rd_vld_d;
        end
    end

    assign l3h_we    = (st_q==S_WRITE) && ((batch_q*P + wr_q) < `DIM_L3H);
    assign l3h_waddr = (batch_q*P) + wr_q;
    assign l3h_wdata = mac_out_vec[wr_q];

    assign layer_done_pulse = (st_q==S_WRITE) && (wr_q==P-1) && (batch_q==BATCHN-1);

    // 调试打印：验证首尾样本地址
    always @(posedge clk) begin
        if ((st_q == S_COMP) && start_for_mac)
            $display("[%0t] L3H MAC start, first feature addr=%0d data=%0d",
                     $time, l2_rd_addr, l2_rd_data);
        if ((st_q == S_COMP) && rd_vld_q && (mac_q == `DIM_L2-1))
            $display("[%0t] L3H MAC last  feature addr=%0d data=%0d",
                     $time, l2_rd_addr, l2_rd_data);
    end

endmodule
