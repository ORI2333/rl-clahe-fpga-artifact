// ============================================================================
// 宽 ROM（$readmemh），每行输出 P*16 位，适配 64 路并行权重
// DEPTH = 输入通道数（例如 256），WIDTH = 16*P（例如 1024）
// 每行代表：同一输入通道对应 P 个并行神经元的权重拼接
// ============================================================================
module rom_wide_hex #(
    parameter string HEX_PATH = "F:/EngineeringWarehouse/ISP/RL/2.FPGA/MLP_V1.1/hex/l2_w_b0.hex",
    parameter int    DEPTH    = 256,
    parameter int    WIDTH    = 1024, // 64*16
    parameter int    ADDR_WIDTH= $clog2(DEPTH)
)(
    input  logic               clka,
    input  logic [ADDR_WIDTH-1:0] addra,
    output logic [WIDTH-1:0]   douta
);
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(HEX_PATH, mem);
    end

    always_ff @(posedge clka) begin
        douta <= mem[addra];
    end
endmodule
