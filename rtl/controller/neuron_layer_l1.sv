`timescale 1ns/1ps
`include "defines.svh"

module neuron_layer_l1(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         start_pulse,
    input  logic signed [15:0]           feature_in [`DIM_IN],   // 7D向量

    // 写 L1 RAM
    output logic                         ram_we,
    output logic [$clog2(`DIM_L1)-1:0]   ram_waddr,
    output logic signed [15:0]           ram_wdata,

    output logic                         layer_done_pulse
);
    localparam int P      = `P_WIDTH;                 // 64
    localparam int BATCHN = `DIM_L1 / `P_WIDTH; // 128 / 64 = 2

    `include "F:/EngineeringWarehouse/ISP/RL/2.FPGA/MLP_V2.0/rtl/verilog_headers/layer1_weights.svh"
    `include "F:/EngineeringWarehouse/ISP/RL/2.FPGA/MLP_V2.0/rtl/verilog_headers/layer1_biases.svh"

typedef enum logic [1:0] {S_IDLE,S_COMP,S_LATCH,S_WRITE} st_t; // + S_LATCH
    st_t st;

    logic [$clog2(BATCHN)-1:0] batch_cnt;
    logic [$clog2(P)-1:0]      wr_cnt;

    // 64 路并行 neuron（一次有效脉冲）
    logic [P-1:0]         v_out_vec;    // 打包向量
    logic signed [15:0]   y_out   [P];
    logic                 comp_start;
    
// 每个并行 neuron 对应的权重索引
logic [$clog2(`DIM_L1)-1:0] neuron_idx_arr [P];

// 组合逻辑：根据 batch_cnt 计算每个并行 neuron 的索引
always_comb begin
    for (int k = 0; k < P; k++) begin
        neuron_idx_arr[k] = batch_cnt * P + k[$clog2(P)-1:0];
        // 理论上 batch_cnt = 0/1 时，对应 0..63, 64..127
    end
end
 // 64 路并行 neuron
genvar i;
generate
    for (i = 0; i < P; i++) begin : G_NEUR
        neuron #(
        .INPUT_COUNT                       (`DIM_IN                        ),
        .ENABLE_RELU                       (1'b1                           ) 
        ) u_neuron (
        .clk                               (clk                            ),
        .rst_n                             (rst_n                          ),
        .data_valid_in                     (comp_start                     ),
        .feature_in                        (feature_in                     ),
        .weights_in                        (LAYER1_WEIGHTS[neuron_idx_arr[i]]),
        .bias_in                           (LAYER1_BIASES[neuron_idx_arr[i]]),
        .data_valid_out                    (v_out_vec[i]                   ),
        .result_out                        (y_out[i]                       ) 
        );
    end
endgenerate

    wire comp_done = &v_out_vec;        // 合法归约


    // ① comp_done打一拍，避免&归约毛刺
logic comp_done_q;
always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) comp_done_q <= 1'b0;
    else       comp_done_q <= comp_done;
end
    // ② 结果锁存阵列
logic signed [15:0] y_latch [P];

// ③ 状态机
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        st <= S_IDLE; batch_cnt <= '0; wr_cnt <= '0; comp_start <= 1'b0;
    end else begin
        comp_start <= 1'b0;
        unique case (st)
            S_IDLE: if (start_pulse) begin
                st <= S_COMP; batch_cnt <= '0; wr_cnt <= '0; comp_start <= 1'b1;
            end

            S_COMP: if (comp_done_q) begin
                // 进入锁存态，下一拍统一锁存
                st <= S_LATCH;
            end

            S_LATCH: begin
                // ④ 同一拍把所有y_out锁存到本地
                integer k;
                for (k=0; k<P; k++) begin
                    y_latch[k] <= y_out[k];
                end
                // 下一拍进入写
                st <= S_WRITE;
                wr_cnt <= '0;
            end

            S_WRITE: begin
                if (wr_cnt == P-1) begin
                    if (batch_cnt == BATCHN-1) begin
                        st <= S_IDLE;
                    end else begin
                        st <= S_COMP; batch_cnt <= batch_cnt + 1; comp_start <= 1'b1;
                    end
                    wr_cnt <= '0;
                end else begin
                    wr_cnt <= wr_cnt + 1;
                end
            end
        endcase
    end
end


// ⑤ 写 RAM 只在写态，从 y_latch 取数
assign ram_we    = (st==S_WRITE);
assign ram_waddr = (batch_cnt*P) + wr_cnt;
assign ram_wdata = y_latch[wr_cnt];

assign layer_done_pulse = (st==S_WRITE) && (wr_cnt==P-1) && (batch_cnt==BATCHN-1);

// 调试打印保持不变（这下 L1[0] 会是 3759 了）
always @(posedge clk) begin
    if (ram_we && ram_waddr < 8)
        $display("[%0t] L1[%0d] = %0d", $time, ram_waddr, ram_wdata);
end
endmodule
