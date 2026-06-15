`timescale 1ns/1ps
// tanh_q12_lut.v  ——  使用 ../hex/tanh_q12.hex （8193行）
// 输入/输出：Q4.12 有符号
module tanh_q12_lut (
    input  wire         clk,
    input  wire signed [15:0] in_q12,   // raw actor y_q12
    output reg  signed [15:0] out_q12   // tanh(y)_q12
);
    // [-4096, +4096] → [0, 8192]
    wire signed [15:0] in_sat =
        (in_q12 > 16'sd4096)  ? 16'sd4096  :
        (in_q12 < -16'sd4096) ? -16'sd4096 : in_q12;
    wire [13:0] addr = in_sat + 16'sd4096;  // 14bit 足够 0..8192

    // ROM: 8193 x 16
    (* rom_style = "block" *) reg [15:0] lut [0:8192];
    initial $readmemh("hex/tanh_q12.hex", lut);

    always @(posedge clk) begin
        out_q12 <= lut[addr];
    end
endmodule
