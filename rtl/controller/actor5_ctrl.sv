`timescale 1ns/1ps
`include "defines.svh"

// 5步 ΔCL 闭环（Q4.12），适配学生 MLP: 5 -> 128 -> 64 -> 1
module actor5_ctrl(
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start_pulse,
    // 这里 DIM_IN = 5： [mu, var, H2, CL_norm, step_norm]
    input  logic signed [15:0] obs_in [`DIM_IN],

    output logic               final_valid,
    output logic signed [15:0] final_cl_q12
);
    localparam int STEPS = 5;
    localparam int FRAC  = `Q_FRAC_BITS;           // 一般 12

    // ---------------- 与 Python 保持一致的标度 ----------------
    // Python 训练脚本中：DELTA_CL_MAX = 2.0
    // 这里直接用 Q4.12 表示 2.0
    localparam logic signed [15:0] DELTA_CL_MAX_Q12 = 16'sd8192;   // 2.0 * 4096

    // CL 在本模块中用“归一化域”表示，范围约 [0, 1]
    // 与 Python 的物理 CL∈[0,10] 对应关系：CL_norm = CL/10
    // 因此这里的 clip 取 [0.0, 1.0]
    localparam logic signed [15:0] CL_MIN_Q12 = 16'sd0;      // 0.0
    localparam logic signed [15:0] CL_MAX_Q12 = 16'sd4096;   // 1.0

    // 工作副本
    logic signed [15:0] obs    [`DIM_IN];
    logic signed [15:0] cl_q12;

    // MLP 接口
    logic               mlp_start;
    logic               mlp_y_valid;
    logic signed [15:0] mlp_y_q12;

    // 学生版 MLP 顶层：5 -> 128 -> 64 -> 1
    mlp_student_top_2h u_mlp (
        .clk         (clk),
        .rst_n       (rst_n),
        .start_pulse (mlp_start),
        .obs_q12     (obs),
        .y_valid     (mlp_y_valid),
        .y_q12       (mlp_y_q12)
    );

    // ---------------- 工具函数 ----------------

    // 16bit 饱和
    function automatic logic signed [15:0] sat16(input logic signed [31:0] x);
        if      (x > 32'sd32767)  return 16'sd32767;
        else if (x < -32'sd32768) return 16'sd32768;
        else                      return x[15:0];
    endfunction

    // Q4.12 乘法：返回 Q4.12（a*b >> FRAC）
    function automatic logic signed [15:0] q12_mul(
        input logic signed [15:0] a,
        input logic signed [15:0] b
    );
        logic signed [31:0] t;  // Q8.24
        t = $signed(a) * $signed(b);
        return sat16(t >>> FRAC);
    endfunction

    // 对 CL 做 clip：归一化域 [0,1] → Q4.12 [0,4096]
    function automatic logic signed [15:0] sat_cl_q12(input logic signed [31:0] x_q12);
        if      (x_q12 < CL_MIN_Q12) return CL_MIN_Q12;
        else if (x_q12 > CL_MAX_Q12) return CL_MAX_Q12;
        else                         return x_q12[15:0];
    endfunction

    // PWL-tanh 近似（沿用你原来的实现）
    function automatic logic signed [15:0] tanh_q12(input logic signed [15:0] x);
        logic signed [15:0] ax;  // |x|
        logic        sign_neg;
        logic signed [31:0] y;

        // 断点、斜率、截距（Q4.12）
        localparam logic signed [15:0] BP_0p5 = 16'sd2048;   // 0.5
        localparam logic signed [15:0] BP_1p5 = 16'sd6144;   // 1.5
        localparam logic signed [15:0] BP_3p0 = 16'sd12288;  // 3.0
        localparam logic signed [15:0] BP_4p0 = 16'sd16384;  // 4.0

        localparam logic signed [15:0] M1 = 16'sd3784;
        localparam logic signed [15:0] M2 = 16'sd1815;
        localparam logic signed [15:0] B2 = 16'sd985;
        localparam logic signed [15:0] M3 = 16'sd245;
        localparam logic signed [15:0] B3 = 16'sd3342;
        localparam logic signed [15:0] M4 = 16'sd18;
        localparam logic signed [15:0] B4 = 16'sd4020;
        localparam logic signed [15:0] ONE_Q12 = 16'sd4096;

        sign_neg = x[15];
        ax       = sign_neg ? -x : x;

        if (ax <= BP_0p5)       y = ($signed(M1)*$signed(ax)) >>> FRAC;
        else if (ax <= BP_1p5)  y = (($signed(M2)*$signed(ax)) >>> FRAC) + $signed(B2);
        else if (ax <= BP_3p0)  y = (($signed(M3)*$signed(ax)) >>> FRAC) + $signed(B3);
        else if (ax <= BP_4p0)  y = (($signed(M4)*$signed(ax)) >>> FRAC) + $signed(B4);
        else                    y = ONE_Q12;

        tanh_q12 = sign_neg ? -y[15:0] : y[15:0];
    endfunction


    logic   signed [              15: 0]     y_raw_q12                        ;
    logic   signed [              15: 0]     y_tanh_q12                       ;
    logic   signed [              15: 0]     dcl_q12                          ;
    logic   signed [              15: 0]     cl_next_q12                      ;
    logic   signed [              31: 0]     cl_sum_q12                       ;

    

    // ---------------- 状态机 ----------------
    typedef enum logic [2:0] {
        S_IDLE, S_LOAD, S_PREP, S_LAUNCH, S_WAIT, S_ACC, S_NEXT, S_DONE
    } st_t;

    st_t          st, st_n;
    logic [2:0]   step_cnt;

    // 组合：状态转移 + 控制信号
    always_comb begin
        mlp_start   = 1'b0;
        final_valid = 1'b0;
        st_n        = st;

        unique case (st)
            S_IDLE  : if (start_pulse) st_n = S_LOAD;
            S_LOAD  : st_n = S_PREP;
            S_PREP  : st_n = S_LAUNCH;
            S_LAUNCH: begin mlp_start = 1'b1; st_n = S_WAIT; end
            S_WAIT  : if (mlp_y_valid) st_n = S_ACC;
            S_ACC   : st_n = S_NEXT;
            S_NEXT  : st_n = (step_cnt == STEPS-1) ? S_DONE : S_LAUNCH;
            S_DONE  : begin final_valid = 1'b1; st_n = S_IDLE; end
            default : st_n = S_IDLE;
        endcase
    end

    // 时序：寄存器、CL 闭环
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st       <= S_IDLE;
            step_cnt <= '0;
            cl_q12   <= '0;
            for (int i = 0; i < `DIM_IN; i++) obs[i] <= '0;
        end else begin
            st <= st_n;

            case (st)
                S_IDLE: begin
                    // nothing
                end

                S_LOAD: begin
                    // obs 工作副本
                    for (int i = 0; i < `DIM_IN; i++) obs[i] <= obs_in[i];
                    // CL 仍然取第 3 维（[mu,var,H2,CL,step]）
                    cl_q12   <= obs_in[3];
                    step_cnt <= '0;
                end

                S_PREP: begin
                    // 如需对 step_norm 做预处理，可以放这里
                end

                S_LAUNCH,
                S_WAIT: begin
                    // 等待 MLP 输出
                end

                S_ACC: begin
                     // y_raw_float = mlp_y_q12 / 4096
                    // tanh_float  = tanh(y_raw_float)
                    // dCL_float   = tanh_float * 2.0
                    // 这里 mlp_y_q12/tanh_q12/ΔCL 都用 Q4.12


                    y_raw_q12  = mlp_y_q12;                   // Q4.12
                    y_tanh_q12 = tanh_q12(y_raw_q12);         // Q4.12 ≈ tanh(y_raw)
                    dcl_q12    = q12_mul(y_tanh_q12,
                                         DELTA_CL_MAX_Q12);   // Q4.12 ≈ ΔCL

                    // CL 更新：cl_q12 + dcl_q12，只做 16bit 饱和，不再强行夹 [0,1]
                    cl_sum_q12  = $signed(cl_q12) + $signed(dcl_q12); // Q4.12
                    cl_next_q12 = sat16(cl_sum_q12);                   // 防止 16bit 溢出

                    cl_q12  <= cl_next_q12;
                    obs[3]  <= cl_next_q12;  // 更新到 obs[3]，作为下一步的输入

                    $display("[%0t] step=%0d, y_raw=%0d, tanh(y)=%0d, dCL=%0d, cl_next=%0d",
                             $time, step_cnt, y_raw_q12, y_tanh_q12, dcl_q12, cl_next_q12);
                end

                S_NEXT: begin
                    step_cnt <= step_cnt + 3'd1;
                end

                S_DONE: begin
                    // final_valid 在组合逻辑中已经拉高，这里不再改
                end

                default: ;
            endcase
        end
    end

    assign final_cl_q12 = cl_q12;

endmodule
