// ============================================================================
// 窄 ROM（$readmemh），每行 16bit，用于偏置或输出层权重
// ============================================================================
module small_rom #(
    parameter string HEX_PATH = "hex/l3_w.hex",
    parameter int    DEPTH    = 256,
    parameter int    ADDR_WIDTH= $clog2(DEPTH)
)(
    input  logic               clka,
    input  logic [ADDR_WIDTH-1:0] addra,
    output logic signed [15:0] douta
);
    logic signed [15:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(HEX_PATH, mem);
    end

    always_ff @(posedge clka) begin
        douta <= mem[addra];
    end
endmodule
