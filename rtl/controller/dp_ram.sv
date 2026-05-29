// ============================================================================
// ????? RAM????????/?????
// DEPTH?ADDR_WIDTH ??????DATA ?? 16bit?Q4.12?
// ============================================================================
module dp_ram #(
    parameter int DEPTH = 256,
    parameter int ADDR_WIDTH = $clog2(DEPTH)
)(
    input  logic                 clka,
    input  logic                 wea,
    input  logic [ADDR_WIDTH-1:0] addra,
    input  logic signed [15:0]   dina,

    input  logic                 clkb,
    input  logic [ADDR_WIDTH-1:0] addrb,
    output logic signed [15:0]   doutb
);
    logic signed [15:0] mem [0:DEPTH-1];

    always_ff @(posedge clka) begin
        if (wea) mem[addra] <= dina;
    end
    always_ff @(posedge clkb) begin
        doutb <= mem[addrb];
    end
endmodule
