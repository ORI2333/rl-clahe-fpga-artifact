`timescale 1ns/1ps
`include "defines.svh"

// L2: 256 -> 256
module neuron_layer_l2(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         start_pulse,

    // L1 RAM 读口
    output logic [$clog2(`DIM_L1)-1:0]   l1_rd_addr,
    input  logic signed [15:0]           l1_rd_data,

    // L2 RAM 写口
    output logic                         l2_we,
    output logic [$clog2(`DIM_L2)-1:0]   l2_waddr,
    output logic signed [15:0]           l2_wdata,

    output logic                         layer_done_pulse
);
    localparam int P      = `P_WIDTH;   // 64
    localparam int BATCHN = `DIM_L2/P;  // 4
    localparam int MACN   = `DIM_L1;    // 256

    typedef enum logic [1:0] {S_IDLE,S_COMP,S_WAIT,S_WRITE} st_t;
    st_t st_q, st_d;

    logic [$clog2(BATCHN)-1:0] batch_q, batch_d; // 0..3
    logic [$clog2(MACN)-1:0]   mac_q,   mac_d;   // 0..255
    logic [$clog2(P)-1:0]      wr_q,    wr_d;    // 0..63

    // 宽权重/偏置
    logic [P*16-1:0] w_bus [BATCHN];
    logic [P*16-1:0] b_bus [BATCHN];

    rom_wide_hex #(.HEX_PATH("hex/l2_w_b0.hex"), .DEPTH(`DIM_L1), .WIDTH(P*16)) u_w0(.clka(clk), .addra(mac_q), .douta(w_bus[0]));
    rom_wide_hex #(.HEX_PATH("hex/l2_w_b1.hex"), .DEPTH(`DIM_L1), .WIDTH(P*16)) u_w1(.clka(clk), .addra(mac_q), .douta(w_bus[1]));
    rom_wide_hex #(.HEX_PATH("hex/l2_w_b2.hex"), .DEPTH(`DIM_L1), .WIDTH(P*16)) u_w2(.clka(clk), .addra(mac_q), .douta(w_bus[2]));
    rom_wide_hex #(.HEX_PATH("hex/l2_w_b3.hex"), .DEPTH(`DIM_L1), .WIDTH(P*16)) u_w3(.clka(clk), .addra(mac_q), .douta(w_bus[3]));

    // 偏置HEX按“每行64个偏置拼接且各行相同”的生成法：用 wr_q 做地址也可
    rom_wide_hex #(.HEX_PATH("hex/l2_b_b0.hex"), .DEPTH(P), .WIDTH(P*16), .ADDR_WIDTH($clog2(P))) u_b0(.clka(clk), .addra(wr_q), .douta(b_bus[0]));
    rom_wide_hex #(.HEX_PATH("hex/l2_b_b1.hex"), .DEPTH(P), .WIDTH(P*16), .ADDR_WIDTH($clog2(P))) u_b1(.clka(clk), .addra(wr_q), .douta(b_bus[1]));
    rom_wide_hex #(.HEX_PATH("hex/l2_b_b2.hex"), .DEPTH(P), .WIDTH(P*16), .ADDR_WIDTH($clog2(P))) u_b2(.clka(clk), .addra(wr_q), .douta(b_bus[2]));
    rom_wide_hex #(.HEX_PATH("hex/l2_b_b3.hex"), .DEPTH(P), .WIDTH(P*16), .ADDR_WIDTH($clog2(P))) u_b3(.clka(clk), .addra(wr_q), .douta(b_bus[3]));

    wire [P*16-1:0] w_sel = w_bus[batch_q];
    wire [P*16-1:0] b_sel = b_bus[batch_q];

    // 64 路并行 MAC
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

            neuron_mac #(.INPUT_COUNT(`DIM_L1), .ENABLE_RELU(1'b1)) u_mac (
                .clk       (clk),
                .rst_n     (rst_n),
                .start_in  (start_for_mac),        // 修改：使用对齐脉冲
                .feature_in(l1_rd_data),
                .weight_in (w_i),
                .bias_in   (b_i),
                .done_out  (mac_done_vec[i]),
                .result_out(mac_out_vec[i])
            );
        end
    endgenerate

    // L1 串行读取地址
    assign l1_rd_addr = mac_q;

    // —— 组合：次态 ——
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
                    if (mac_q == `DIM_L1-1) begin
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

    // —— 时序寄存 —— 
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st_q    <= S_IDLE;
            batch_q <= '0;
            mac_q   <= '0;
            wr_q    <= '0;
            rd_vld_q <= 1'b0;
        end else begin
            st_q    <= st_d;
            batch_q <= batch_d;
            mac_q   <= mac_d;
            wr_q    <= wr_d;
            rd_vld_q <= rd_vld_d;
        end
    end

    // 写 L2 RAM（只在写阶段）
    assign l2_we    = (st_q==S_WRITE);
    assign l2_waddr = (batch_q*P) + wr_q;
    assign l2_wdata = mac_out_vec[wr_q];

    // 层完成
    assign layer_done_pulse = (st_q==S_WRITE) && (wr_q==P-1) && (batch_q==BATCHN-1);

    // 调试打印：验证首尾样本地址
    always @(posedge clk) begin
        if ((st_q == S_COMP) && start_for_mac)
            $display("[%0t] L2 MAC start, first feature addr=%0d data=%0d",
                     $time, l1_rd_addr, l1_rd_data);
        if ((st_q == S_COMP) && rd_vld_q && (mac_q == `DIM_L1-1))
            $display("[%0t] L2 MAC last  feature addr=%0d data=%0d",
                     $time, l1_rd_addr, l1_rd_data);
    end

endmodule
