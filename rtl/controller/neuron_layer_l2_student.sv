`timescale 1ns/1ps
`include "defines.svh"

// 学生版 L2: DIM_L1(128) -> DIM_L2(64)
module neuron_layer_l2_student(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        start_pulse,

    // L1 RAM 读口
    output logic [$clog2(`DIM_L1)-1:0]  l1_rd_addr,
    input  logic signed [15:0]          l1_rd_data,

    // L2 RAM 写口
    output logic                        l2_we,
    output logic [$clog2(`DIM_L2)-1:0]  l2_waddr,
    output logic signed [15:0]          l2_wdata,

    output logic                        layer_done_pulse
);
    localparam int P      = `P_WIDTH;      // 64
    localparam int NIN    = `DIM_L1;      // 128
    localparam int NOUT   = `DIM_L2;      // 64
    localparam int BATCHN = (NOUT + P - 1)/P; // 1

    typedef enum logic [1:0] {S_IDLE,S_COMP,S_WAIT,S_WRITE} st_t;
    st_t st_q, st_d;

    logic [$clog2(BATCHN)-1:0] batch_q, batch_d;
    logic [$clog2(NIN)-1:0]    mac_q,   mac_d;
    logic [$clog2(P)-1:0]      wr_q,    wr_d;

    // 宽权重/偏置总线（只有一组）
    logic [P*16-1:0] w_bus;
    logic [P*16-1:0] b_bus;

    // 根据自己的工程路径改这里
    rom_wide_hex #(
        .HEX_PATH   ("F:/EngineeringWarehouse/ISP/RL/4.SAC/FPGA/hex/l2_w_b0.hex"),
        .DEPTH      (NIN),
        .WIDTH      (P*16)
    ) u_w0 (
        .clka (clk),
        .addra(mac_q),
        .douta(w_bus)
    );

    rom_wide_hex #(
        .HEX_PATH   ("F:/EngineeringWarehouse/ISP/RL/4.SAC/FPGA/hex/l2_b_b0.hex"),
        .DEPTH      (P),
        .WIDTH      (P*16),
        .ADDR_WIDTH ($clog2(P))
    ) u_b0 (
        .clka (clk),
        .addra(wr_q),
        .douta(b_bus)
    );

    // 选择 batch（这里 BATCHN=1，其实就是 w_bus/b_bus）
    wire [P*16-1:0] w_sel = w_bus;
    wire [P*16-1:0] b_sel = b_bus;

    // 64 路并行 MAC
    logic [P-1:0]       mac_done_vec;
    logic signed [15:0] mac_out_vec [P];

    // 读数据对齐标志（L1 读口 1 拍延迟）
    logic rd_vld_q, rd_vld_d;
    wire  start_for_mac = (st_q == S_COMP) && rd_vld_q && (mac_q == '0);

    genvar i;
    generate
        for (i=0;i<P;i++) begin: G_MAC
            wire signed [15:0] w_i = w_sel[(i+1)*16-1 -: 16];
            wire signed [15:0] b_i = b_sel[(i+1)*16-1 -: 16];

            neuron_mac #(
                .INPUT_COUNT (NIN),
                .ENABLE_RELU (1'b1)
            ) u_mac (
                .clk        (clk),
                .rst_n      (rst_n),
                .start_in   (start_for_mac),
                .feature_in (l1_rd_data),
                .weight_in  (w_i),
                .bias_in    (b_i),
                .done_out   (mac_done_vec[i]),
                .result_out (mac_out_vec[i])
            );
        end
    endgenerate

    assign l1_rd_addr = mac_q;

    // 组合逻辑：状态机下一状态
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
                    rd_vld_d = 1'b0;
                end
            end

            S_COMP: begin
                if (!rd_vld_q) begin
                    // 首拍对齐
                    rd_vld_d = 1'b1;
                    mac_d    = '0;
                end else begin
                    // 真正喂 0..NIN-1 个输入
                    if (mac_q == NIN-1)
                        st_d = S_WAIT;
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
                    // NOUT = 64, P = 64, 只写 64 个输出
                    st_d = S_IDLE;
                    wr_d = '0;
                end else begin
                    wr_d = wr_q + 1'b1;
                end
            end

            default: st_d = S_IDLE;
        endcase
    end

    // 时序寄存
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

    // 写 L2 RAM：只有 0..63 有效
    assign l2_we    = (st_q==S_WRITE) && (wr_q < NOUT);
    assign l2_waddr = wr_q;
    assign l2_wdata = mac_out_vec[wr_q];

    assign layer_done_pulse = (st_q==S_WRITE) && (wr_q==NOUT-1);

endmodule
