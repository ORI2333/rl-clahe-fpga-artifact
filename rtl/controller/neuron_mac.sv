`include "defines.svh"

// MAC block for Q4.12 inputs: accumulates in Q8.24, aligns bias, and outputs saturated Q4.12.
module neuron_mac #(
    parameter int INPUT_COUNT = 256,
    parameter bit ENABLE_RELU = 1'b1
)(
    input  logic               clk,
    input  logic               rst_n,

    // INPUT_COUNT pairs of (feature, weight, bias)
    input  logic               start_in,

    input  logic signed [15:0] feature_in,   // Q4.12
    input  logic signed [15:0] weight_in,    // Q4.12
    input  logic signed [15:0] bias_in,      // Q4.12

    output logic               done_out,     // 1-cycle pulse when result ready
    output logic signed [15:0] result_out    // Q4.12
);
    localparam int FRAC = `Q_FRAC_BITS;

    typedef enum logic [1:0] {S_IDLE, S_ACC, S_DONE} state_t;
    state_t st_q, st_d;

    logic [$clog2(INPUT_COUNT)-1:0] cnt_q, cnt_d;
    logic signed [47:0]             acc_q, acc_d;   // 48位累加器（Q8.24格式！）
    logic signed [15:0]             bias_q, bias_d; // 锁存偏置（Q4.12）

    // 乘法：Q4.12 × Q4.12 -> Q8.24（不立即右移！保持完整精度）
    logic signed [31:0] prod_q24;

    always_comb begin
        prod_q24 = feature_in * weight_in; // 16×16 -> 32位 Q8.24
    end

    function automatic signed [15:0] sat16(input signed [31:0] x);
        if (x >  32'sd32767)   return 16'sh7FFF;
        if (x < -32'sd32768)   return 16'sh8000;
        return x[15:0];
    endfunction

    logic done_d;
    always_comb begin
        st_d   = st_q;
        cnt_d  = cnt_q;
        acc_d  = acc_q;
        bias_d = bias_q;
        done_d = 1'b0;

        case (st_q)
            S_IDLE: begin
                if (start_in) begin
                    st_d   = S_ACC;
                    cnt_d  = '0;
                    acc_d  = '0;
                    bias_d = bias_in;
                end
            end

            S_ACC: begin
                // 累加Q8.24：将32位乘积符号扩展到48位后累加（不右移！）
                acc_d = acc_q + {{16{prod_q24[31]}}, prod_q24};
                if (cnt_q == INPUT_COUNT-1)
                    st_d = S_DONE;
                else
                    cnt_d = cnt_q + 1'b1;
            end

            S_DONE: begin
                done_d = 1'b1;
                st_d   = S_IDLE;
            end

            default: st_d = S_IDLE;
        endcase
    end

    // Align bias to Q8.24, add once, then shift back to Q4.12.
    wire signed [47:0] bias_aligned_q24 = {{32{bias_q[15]}}, bias_q} <<< FRAC;
    wire signed [47:0] sum_q24          = acc_q + bias_aligned_q24;
    wire signed [31:0] sum_q12_32       = sum_q24 >>> FRAC;
    wire signed [15:0] lin_q12          = sat16(sum_q12_32);

    logic done_q;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st_q       <= S_IDLE;
            cnt_q      <= '0;
            acc_q      <= '0;
            bias_q     <= '0;
            done_q     <= 1'b0;
            result_out <= '0;
        end else begin
            st_q   <= st_d;
            cnt_q  <= cnt_d;
            acc_q  <= acc_d;
            bias_q <= bias_d;
            done_q <= done_d;

            if (st_q == S_DONE) begin
                if (!ENABLE_RELU) begin
                    $display("[DBG] acc_q=%0d, bias_q=%0d, lin_q12=%0d", acc_q, bias_q, lin_q12);
                end
                if (ENABLE_RELU && lin_q12[15])
                    result_out <= 16'sd0;
                else
                    result_out <= lin_q12;
            end
        end
    end

    assign done_out = done_q;

endmodule
