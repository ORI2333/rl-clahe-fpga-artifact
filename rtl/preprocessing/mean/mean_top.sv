`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:
// Engineer:
//
// Create Date: 2025/08/13 18:00:00
// Design Name:
// Module Name: mean_top
// Project Name:
// Target Devices: xc7k325
// Tool Versions:
// Description:
// 统计图像的平均值，帧结束后输出平均值。
// 本版本使用 Divider Generator IP Core 进行除法运算。
// Dependencies: div_gen_mean (Divider Generator IP instance)
//
// Revision:
// Revision 2.00 - Integrated Divider Generator IP Core
// Additional Comments:
//
//////////////////////////////////////////////////////////////////////////////////

module mean_top#(
        parameter          PIXEL_WIDTH      = 8                    ,        // 像素位宽
        parameter          IMAGE_WIDTH      = 640                  ,      // 图像宽度
        parameter          IMAGE_HEIGHT     = 480                  // 图像高度
)(
        input  wire                                       clk                 ,// 时钟
        input  wire                                       rst_n               ,// 异步复位，低有效
        input  wire                                       MEAN_enable         ,// 使能信号
        input  wire                                       i_vblank            ,// 场同步信号 (消隐期为低)
        input  wire                                       i_hblank            ,// 行同步信号 (消隐期为低)
        input  wire         [   PIXEL_WIDTH-1: 0]         i_data              ,// 输入像素数据

        output wire                                       o_vblank            ,// 透传场同步信号
        output wire                                       o_hblank            ,// 透传行同步信号
        output wire         [   PIXEL_WIDTH-1: 0]         o_data              ,// 透传像素数据
        output reg          [   PIXEL_WIDTH-1: 0]         o_mean_gray         ,// 输出的平均灰度值
        output reg                                        o_mean_valid         // 平均值有效信号 (单周期脉冲)
);

//==================================================================================
// 1. 参数和内部信号定义
//==================================================================================

// IP核接口位宽 (根据你的模板)
        localparam         IP_DIVISOR_WIDTH = 24                   ;
        localparam         IP_DIVIDEND_WIDTH= 32                   ;

// 状态机状态定义
        localparam         STATE_IDLE       = 3'b001               ; // 空闲状态
        localparam         STATE_ACCUM      = 3'b010               ; // 累加状态
        localparam         STATE_DIVIDE     = 3'b100               ; // 除法状态

// 状态机寄存器
reg            [               2: 0]     current_state          ;

// 累加器和计数器 (根据图像参数精确计算)
        localparam         SUM_WIDTH        = $clog2(IMAGE_WIDTH * IMAGE_HEIGHT * (2**PIXEL_WIDTH - 1)); // 27位
        localparam         COUNT_WIDTH      = $clog2(IMAGE_WIDTH * IMAGE_HEIGHT);                     // 19位
reg            [     SUM_WIDTH-1: 0]     pixel_sum              ;
reg            [   COUNT_WIDTH-1: 0]     pixel_count            ;

// 边沿检测和像素有效信号
reg                                      i_vblank_d1            ;
wire                                     pixel_valid            ;
wire                                     frame_start            ;
wire                                     frame_end              ;

//-----------------------------------
// 与Divider IP核交互的信号
//-----------------------------------
// 传递给IP核的数据寄存器
reg            [IP_DIVIDEND_WIDTH-1: 0]     pixel_sum_to_div       ;
reg            [IP_DIVISOR_WIDTH-1: 0]     pixel_count_to_div     ;

// 控制和状态信号
reg                                      start_divide           ;// 连接到IP核的tvalid输入
wire                                     divide_result_valid    ;// 来自IP核的tvalid输出
wire           [              55: 0]     divide_result          ;// 来自IP核的tdata输出

//==================================================================================
// 2. 逻辑实现
//==================================================================================

// 像素有效和帧同步信号
assign   pixel_valid      = MEAN_enable & i_vblank & i_hblank;
assign   frame_start      = ~i_vblank_d1 & i_vblank;
assign   frame_end        = i_vblank_d1 & ~i_vblank;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) i_vblank_d1 <= 1'b0;
    else        i_vblank_d1 <= i_vblank;
end

// 透传输入视频信号
assign   o_vblank         = i_vblank             ;
assign   o_hblank         = i_hblank             ;
assign   o_data           = i_data               ;



//==================================================================================
// 4. 状态机与数据处理逻辑 (单always块)
//==================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_state      <= STATE_IDLE;
        pixel_sum          <= 0;
        pixel_count        <= 0;
        o_mean_gray        <= 0;
        o_mean_valid       <= 1'b0;
        start_divide       <= 1'b0;
        pixel_sum_to_div   <= 0;
        pixel_count_to_div <= 0;
    end
    else begin
        // --- 每个周期默认行为 ---
        o_mean_valid <= 1'b0;                                       // valid信号默认为低，只在需要时拉高一拍

        case (current_state)
            STATE_IDLE: begin
                start_divide <= 1'b0;                               // 确保在IDLE状态停止发起运算
                if (frame_start && MEAN_enable) begin
                    // 帧开始，进入累加状态
                    current_state <= STATE_ACCUM;
                    pixel_sum     <= 0;
                    pixel_count   <= 0;
                end
            end

            STATE_ACCUM: begin
                // 在有效视频区域内进行累加
                if (pixel_valid) begin
                    pixel_sum   <= pixel_sum + i_data;
                    pixel_count <= pixel_count + 1;
                end

                // 帧结束，进入除法状态
                if (frame_end && MEAN_enable) begin
                    current_state <= STATE_DIVIDE;
                    // 将最终的累加值和计数值锁存，准备送给IP核
                    pixel_sum_to_div   <= {{(IP_DIVIDEND_WIDTH-SUM_WIDTH){1'b0}}, pixel_sum};
                    // 为防止除以0，如果计数值为0，则将除数设为1
                    pixel_count_to_div <= (pixel_count == 0) ? 1 : {{(IP_DIVISOR_WIDTH-COUNT_WIDTH){1'b0}}, pixel_count};
                end
                else if (!MEAN_enable) begin
                    // 如果中途被禁用，直接返回IDLE
                    current_state <= STATE_IDLE;
                end
            end

            STATE_DIVIDE: begin
                // 拉高IP核的valid输入，开始运算
                start_divide <= 1'b1;

                // 等待IP核运算完成
                if (divide_result_valid) begin
                    current_state <= STATE_IDLE;                    // 返回空闲
                    start_divide  <= 1'b0;                          // 停止发起运算
                    o_mean_valid  <= 1'b1;                          // 输出有效信号
                    // IP核输出的数据是{余数, 商}，商在低位
                    // 商的位宽=被除数位宽(32)，我们只需取低8位
                    o_mean_gray   <= divide_result[IP_DIVIDEND_WIDTH-1:  IP_DIVIDEND_WIDTH-PIXEL_WIDTH];
                end
            end

            default: begin
                current_state <= STATE_IDLE;
            end
        endcase
    end
end

//==================================================================================
// 3. Divider IP核实例化
//==================================================================================
div_gen_mean u_div_gen_mean (
        .aclk                              (clk                            ),// input wire aclk
        .s_axis_divisor_tvalid             (start_divide                   ),// input wire s_axis_divisor_tvalid
        .s_axis_divisor_tdata              (pixel_count_to_div             ),// input wire [23 : 0] s_axis_divisor_tdata
        .s_axis_dividend_tvalid            (start_divide                   ),// input wire s_axis_dividend_tvalid
        .s_axis_dividend_tdata             (pixel_sum_to_div               ),// input wire [31 : 0] s_axis_dividend_tdata
        .m_axis_dout_tvalid                (divide_result_valid            ),// output wire m_axis_dout_tvalid
        .m_axis_dout_tdata                 (divide_result                  ) // output wire [55 : 0] m_axis_dout_tdata
);

endmodule