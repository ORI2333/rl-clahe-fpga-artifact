`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/15 15:18:06
// Design Name: 
// Module Name: std_dev_top
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
//   计算一帧图像灰度值的方差 (Variance)。
//   修改版：输出 Q14.8 格式的定点数方差。
//   收紧了小数位宽，避免结果膨胀到 64 位。
// Dependencies: 
// 
//////////////////////////////////////////////////////////////////////////////////

module variance #(
        parameter          PIXEL_WIDTH      = 8                    ,  // 像素位宽
        parameter          IMAGE_WIDTH      = 320                  ,  // 图像宽度
        parameter          IMAGE_HEIGHT     = 240                  // 图像高度
)(
        input  wire                                       clk                 ,
        input  wire                                       rst_n               ,
        input  wire                                       STD_enable          ,
        input  wire                                       i_vblank            ,
        input  wire                                       i_hblank            ,
        input  wire         [   PIXEL_WIDTH-1: 0]         i_data              ,
        output wire                                       o_vblank            ,
        output wire                                       o_hblank            ,
        output wire         [   PIXEL_WIDTH-1: 0]         o_data              ,
        output reg                                        o_variance_valid    ,
        output reg          [              26: 0]         o_variance_int      ,// 整数
        output reg          [              16: 0]         o_variance_frac      // 小数
);

// --- 定点数格式定义 (Q14.8) ---
        localparam         FIXED_INT_BITS   = 14                   ;
        localparam         FIXED_FRAC_BITS  = 8                    ;
        localparam         FIXED_WIDTH      = FIXED_INT_BITS + FIXED_FRAC_BITS; // 22

// --- IP核延迟 ---
        localparam         DIV_LATENCY      = 49 + 1               ;
        localparam         MUL_LATENCY      = 5                    ;

        localparam         SIGNED_REF_K     = 1 << (PIXEL_WIDTH - 1);

        localparam         COUNT_WIDTH      = $clog2(IMAGE_WIDTH * IMAGE_HEIGHT); // 17 for 320x240
        localparam         SUM_Y_WIDTH      = $clog2(IMAGE_WIDTH * IMAGE_HEIGHT * SIGNED_REF_K) + 1; // ~25
        localparam         SUM_Y_SQ_WIDTH   = $clog2(IMAGE_WIDTH * IMAGE_HEIGHT * (SIGNED_REF_K ** 2)); // ~31

// 状态机
        localparam         STATE_IDLE       = 3'b001               ;
        localparam         STATE_ACCUM      = 3'b010               ;
        localparam         STATE_CALC_PIPE  = 3'b011               ;
        localparam         STATE_DONE       = 3'b100               ;

reg            [               2: 0]     current_state          ;
reg            [   COUNT_WIDTH-1: 0]     pixel_count            ;
reg     signed [     SUM_Y_WIDTH: 0]     sum_y                  ;
reg            [SUM_Y_SQ_WIDTH-1: 0]     sum_y_sq               ;

// 除法输入
reg            [SUM_Y_SQ_WIDTH+FIXED_FRAC_BITS-1: 0]     div_dividend_unsigned  ;
reg            [   COUNT_WIDTH-1: 0]     div_divisor_unsigned   ;

// 符号保存
reg                                      mean_y_sign            ;

reg            [   COUNT_WIDTH-1: 0]     r_pixel_count          ;
reg     signed [   SUM_Y_WIDTH: 0]     r_sum_y                ;
reg            [SUM_Y_SQ_WIDTH-1: 0]     r_sum_y_sq             ;
reg            [               7: 0]     calc_step              ;
reg            [               7: 0]     calc_wait_cnt          ;

// IP核接口
reg                                      div_tvalid             ;
reg            [SUM_Y_SQ_WIDTH+FIXED_FRAC_BITS-1: 0]     div_dividend           ;
reg            [   COUNT_WIDTH-1: 0]     div_divisor            ;
wire                                     div_result_tvalid      ;
wire           [   FIXED_WIDTH-1: 0]     div_result_tdata       ;

reg                                      mul_tvalid             ;
reg     signed [   FIXED_WIDTH-1: 0]     mul_a                  ;
reg     signed [   FIXED_WIDTH-1: 0]     mul_b                  ;
reg    signed [ 2*FIXED_WIDTH-1: 0]     mul_result_tdata       ;// 44 位

wire           [              63: 0]     div_result_full        ;

// 中间结果
reg     signed [     FIXED_WIDTH: 0]     mean_y_q               ;// E[y]
reg            [   FIXED_WIDTH-1: 0]     mean_of_y_sq_q         ;// E[y^2]
reg            [ 2*FIXED_WIDTH-1: 0]     mean_y_squared_q       ;
reg     signed [ 2*FIXED_WIDTH-1: 0]     variance_q             ;

// 输出
reg     signed [   2*FIXED_WIDTH: 0]     temp_variance          ;


// input 信号处理
reg                                      i_vblank_d1            ;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) i_vblank_d1 <= 1'b0;
    else        i_vblank_d1 <= i_vblank;
end

wire                                     pixel_valid          =STD_enable && i_vblank && i_hblank;
wire                                     frame_start          =~i_vblank_d1 && i_vblank;
wire                                     frame_end            =i_vblank_d1 && ~i_vblank;

// 透传
assign   o_vblank         = i_vblank             ;
assign   o_hblank         = i_hblank             ;
assign   o_data           = i_data               ;

// Divider 输出对齐
assign   div_result_tdata = div_result_full[FIXED_WIDTH+24-1:24];

// ---------------- Pipeline ----------------
reg                                      pixel_valid_d1,      pixel_valid_d2;
reg     signed [     PIXEL_WIDTH: 0]     y_diff_d1,           y_diff_d2;
reg            [   2*PIXEL_WIDTH: 0]     y_sq_d1                ;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pixel_valid_d1 <= 1'b0;
        pixel_valid_d2 <= 1'b0;
        y_diff_d1      <= 0;
        y_diff_d2      <= 0;
        y_sq_d1        <= 0;
    end else begin
        pixel_valid_d1 <= pixel_valid;
        y_diff_d1      <= {1'b0, i_data} - SIGNED_REF_K;
        pixel_valid_d2 <= pixel_valid_d1;
        y_diff_d2      <= y_diff_d1;
        y_sq_d1        <= y_diff_d1 * y_diff_d1;
    end
end

// ---------------- FSM ----------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_state    <= STATE_IDLE;
        pixel_count      <= 0;
        sum_y            <= 0;
        sum_y_sq         <= 0;
        o_variance_int       <= 0;
        o_variance_frac      <= 0;
        o_variance_valid <= 1'b0;
        calc_step        <= 0;
        calc_wait_cnt    <= 0;
        r_pixel_count    <= 0;
        r_sum_y          <= 0;
        r_sum_y_sq       <= 0;
        mean_y_q         <= 0;
        mean_of_y_sq_q   <= 0;
        mean_y_squared_q <= 0;
        variance_q       <= 0;
        div_tvalid       <= 1'b0;
        mul_tvalid       <= 1'b0;
        mul_a            <= 0;
        mul_b            <= 0;
        temp_variance    <= 0;
        div_dividend_unsigned <= 0;
        div_divisor_unsigned  <= 0;
        mean_y_sign      <= 1'b0;
       // mul_result_tdata <= 0;
    end else begin
        o_variance_valid <= 1'b0;

        case (current_state)
            STATE_IDLE: begin
                if (frame_start && STD_enable) begin
                    current_state <= STATE_ACCUM;
                    pixel_count   <= 0;
                    sum_y         <= 0;
                    sum_y_sq      <= 0;
                end
            end
            STATE_ACCUM: begin
                if (pixel_valid_d2) begin
                    pixel_count <= pixel_count + 1;
                    sum_y       <= sum_y + y_diff_d2;
                    sum_y_sq    <= sum_y_sq + y_sq_d1;
                end
                if (frame_end) begin
                    current_state <= STATE_CALC_PIPE;
                    r_pixel_count <= pixel_count;
                    r_sum_y       <= sum_y;
                    r_sum_y_sq    <= sum_y_sq;
                    calc_step     <= 0;
                end
            end
            STATE_CALC_PIPE: begin
                case (calc_step)
                    0: begin
                        div_tvalid <= 1'b1;
                        if (r_sum_y < 0) begin
                            div_dividend_unsigned <= {-r_sum_y, {FIXED_FRAC_BITS{1'b0}}};
                            mean_y_sign <= 1'b1;
                        end else begin
                            div_dividend_unsigned <= {r_sum_y, {FIXED_FRAC_BITS{1'b0}}};
                            mean_y_sign <= 1'b0;
                        end
                        div_divisor_unsigned <= r_pixel_count;
                        calc_wait_cnt <= DIV_LATENCY - 1;
                        calc_step <= 1;
                    end
                    1: begin
                        div_tvalid <= 1'b0;
                        if (calc_wait_cnt == 0) begin
                            mean_y_q <= mean_y_sign ? -div_result_tdata : div_result_tdata;
                            calc_step <= 2;
                        end else calc_wait_cnt <= calc_wait_cnt - 1;
                    end
                    2: begin
                        div_tvalid <= 1'b1;
                        div_dividend_unsigned <= {r_sum_y_sq, {FIXED_FRAC_BITS{1'b0}}};
                        div_divisor_unsigned <= r_pixel_count;
                        calc_wait_cnt <= DIV_LATENCY - 1;
                        calc_step <= 3;
                    end
                    3: begin
                        div_tvalid <= 1'b0;
                        if (calc_wait_cnt == 0) begin
                            mean_of_y_sq_q <= div_result_tdata;
                            calc_step <= 4;
                        end else calc_wait_cnt <= calc_wait_cnt - 1;
                    end
                    4: begin
                        mul_a <= mean_y_q;
                        mul_b <= mean_y_q;
                        calc_wait_cnt <= MUL_LATENCY - 1;
                        calc_step <= 5;
                    end
                    5: begin
                        mul_tvalid <= 1'b0;
                        if (calc_wait_cnt == 0) begin
                            mean_y_squared_q <= mul_result_tdata;
                            temp_variance <= {mean_of_y_sq_q, {FIXED_FRAC_BITS{1'b0}}} - mul_result_tdata;
                            //variance_q <= temp_variance[2*FIXED_WIDTH-1] ? 0 : temp_variance;
                            current_state <= STATE_DONE;
                            calc_step <= 0;
                        end else calc_wait_cnt <= calc_wait_cnt - 1;
                    end
                endcase
            end
            STATE_DONE: begin
                o_variance_int <= temp_variance[42:(FIXED_FRAC_BITS*2)] ;
                o_variance_frac <= temp_variance[(FIXED_FRAC_BITS*2) -1:0];
                o_variance_valid <= 1'b1;
                current_state <= STATE_IDLE;
            end
        endcase
    end
end

// Divider IP
div_gen_std u_div_gen_std (
        .aclk                              (clk                            ),
        .aclken                            (1'b1                           ),
        .aresetn                           (rst_n                          ),
        .s_axis_divisor_tvalid             (div_tvalid                     ),
        .s_axis_divisor_tdata              ({{(24-COUNT_WIDTH){1'b0}}, div_divisor_unsigned}),
        .s_axis_dividend_tvalid            (div_tvalid                     ),
        .s_axis_dividend_tdata             ({{(48-(SUM_Y_SQ_WIDTH+FIXED_FRAC_BITS)){1'b0}}, div_dividend_unsigned}),
        .m_axis_dout_tvalid                (div_result_tvalid              ),
        .m_axis_dout_tdata                 (div_result_full                ) 
);

// Multiplier IP
mult_gen_std u_mult_gen_std (
        .CLK                               (clk                            ),
        .A                                 (mul_a                          ),// 22 bit
        .B                                 (mul_b                          ),// 22 bit
        .P                                 (mul_result_tdata               ) // 44 bit
);

endmodule
