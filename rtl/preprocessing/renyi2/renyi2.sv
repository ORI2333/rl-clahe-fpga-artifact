`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/26 20:26:40
// Design Name: 
// Module Name: renyi2
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 1. Instantiates the histogram statistics module `entropy_histogram`.
// 2. After statistics are complete, starts a pipeline to calculate Rényi-2 entropy.
// 3. This version uses a purely unsigned datapath for simplicity and efficiency.
// Dependencies: entropy_histogram.v, log2cordic.v
// 
// Revision:
// Revision 3.0 - Refactored to use a purely unsigned calculation datapath.
//
//////////////////////////////////////////////////////////////////////////////////


module renyi2 #(
        parameter          PIXEL_WIDTH      = 8                    ,
        parameter          BLOCK_WIDTH      = 320                  ,
        parameter          BLOCK_HEIGHT     = 240                  ,
        parameter          FIXED_WIDTH      = 32                   //FIXED_WIDTH     = FIXED_INT_BITS + FIXED_FRAC_BITS; // 32
)(
        input  wire                                       clk                 ,
        input  wire                                       rst_n               ,
        input  wire                                       i_hblank            ,// Row active signal (active high)
        input  wire                                       i_vblank            ,// Frame active signal (active high)
        input  wire         [   PIXEL_WIDTH-1: 0]         i_data              ,
        input  wire                                       renyi2_enable         ,

        output reg                                        entropy_valid       ,// Entropy valid signal (single-cycle pulse)
        output reg          [   FIXED_WIDTH-1: 0]         entropy_renyi2       // Output Rényi-2 entropy (Unsigned Q16.16)
);

//==================================================================================
// 1. Parameters and Internal Signals
//==================================================================================
    // --- Fixed-point format definition (Q16.16) ---
        localparam         FIXED_INT_BITS   = 16                   ;
        localparam         FIXED_FRAC_BITS  = 16                   ;
    //localparam          FIXED_WIDTH     = FIXED_INT_BITS + FIXED_FRAC_BITS; // 32

        localparam         LOG2_INPUT_WIDTH = PIXEL_WIDTH*3        ; // Needs to be wide enough for sum_of_squares

    // --- Module Latencies (Please adjust to match your modules) ---
        localparam         MUL_LATENCY      = 3                    ;  
        localparam         LOG2_LATENCY     = 16                    ;  // Assumed to be equal to STAGES of log2cordic

    // --- Counter bit widths ---
        localparam         COUNT_WIDTH      = $clog2(BLOCK_WIDTH * BLOCK_HEIGHT); // 17
        localparam         HIST_DEPTH       = 1 << PIXEL_WIDTH     ;                   // 256

        localparam         PIXEL_COUNT_N    = BLOCK_WIDTH * BLOCK_HEIGHT;
        localparam         LOG2_320x240_INT = 16'd16               ;
        localparam         LOG2_320x240_FRAC= 16'd22881               ; // 0.22881 in Q0.16
        localparam         LOG2_N_Q         = {LOG2_320x240_INT    ,LOG2_320x240_FRAC};  

    // --- State machine definitions ---
        localparam         STATE_IDLE       = 4'h0                 ;
        localparam         STATE_CALC_SUM_SQ= 4'h1                 ; // Calculate sum of squares
        localparam         STATE_CALC_LOG_N = 4'h2                 ; // Calculate log2(N)
        localparam         STATE_WAIT_LOG_N = 4'h3                 ; // Wait for log2(N)
        localparam         STATE_CALC_LOG_SUM_SQ= 4'h4                 ; // Calculate log2(?c²)
        localparam         STATE_WAIT_LOG_SUM_SQ= 4'h5                 ; // Wait for log2(?c²)
        localparam         STATE_DONE       = 4'h6                 ; // Done

reg            [               3: 0]     current_state          ;

    // --- Wires ---
// histogram wires
wire                                     his_valid_results      ;
wire           [   PIXEL_WIDTH-1: 0]     his_data               ;
wire           [              15: 0]     his_data_n             ;


    // --- Internal Registers ---
reg            [   COUNT_WIDTH-1: 0]     pixel_count_N          ;// Total pixel count N
reg            [   PIXEL_WIDTH: 0]     hist_addr_cnt          ;

// multiplication pipeline registers
reg            [              31: 0]     mult_gen_square_P      ;


    // Calculation pipeline registers
reg            [              34: 0]     sum_of_squares         ;// ?count(i)² accumulator

wire           [   COUNT_WIDTH-1: 0]     count_dly              ;


    // Log2 module interface signals
    
reg            [   FIXED_WIDTH-1: 0]     log2_data_i            ;
reg                                      log2_valid_i           ;
wire           [   FIXED_WIDTH-1: 0]     log2_log2_o            ;// Unsigned fixed-point from your module
wire                                     log2_valid_o           ;

    // Latched logarithm results (now unsigned)
reg            [   FIXED_WIDTH-1: 0]     log2_N_q               ;
reg            [   FIXED_WIDTH-1: 0]     log2_sum_sq_q          ;
    
wire                                     pixel_valid          =i_hblank && i_vblank;
reg                                      i_vblank_d1            ;
wire                                     frame_end            =i_vblank_d1 && ~i_vblank;


//==================================================================================
// 3. Core Logic
//==================================================================================
// delay
always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
                i_vblank_d1 <= 1'b0; 
        end else begin
                i_vblank_d1 <= i_vblank;
        end
end
    


// --- Calculation Pipeline State Machine ---
always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
                current_state   <= STATE_IDLE;
                entropy_valid   <= 1'b0;
                entropy_renyi2  <= 0;
                hist_addr_cnt   <= 0;
                sum_of_squares  <= 0;
                log2_valid_i    <= 1'b0;
                log2_N_q        <= 0;
                log2_sum_sq_q   <= 0;
                log2_data_i     <= 0;
        end else begin
   

                case (current_state)
                STATE_IDLE: begin
                                entropy_valid <= 1'b0;                                  // Default to low
                                entropy_renyi2 <= 0;
                                log2_valid_i  <= 1'b0;
                                if (his_valid_results && renyi2_enable) begin
                                                current_state  <= STATE_CALC_SUM_SQ;
                                                hist_addr_cnt  <= 0;
                                                sum_of_squares <= 0;
                                                log2_data_i     <= 0;
                                end
                end
                // calculate sum of squares
                // ??3??????
                STATE_CALC_SUM_SQ: begin
                                if (his_data == 'd0) begin
                                        if (hist_addr_cnt == HIST_DEPTH-1+3) begin
                                                current_state <= STATE_CALC_LOG_SUM_SQ;
                                        end else begin
                                                hist_addr_cnt <= hist_addr_cnt + 1'b1;
                                        end
                                end else begin
                                        sum_of_squares <= sum_of_squares + mult_gen_square_P;
                                        hist_addr_cnt <= his_data;
                                end
                end

                // calculate log2(N) in advance
                // STATE_CALC_LOG_N: begin
                //         log2_valid_i  <= 1'b1;
                //         // Convert integer N to fixed-point format
                //         log2_data_i   <= {pixel_count_N, {FIXED_FRAC_BITS{1'b0}}};
                //         current_state <= STATE_WAIT_LOG_N;
                // end
                
                // STATE_WAIT_LOG_N: begin
                //         if (log2_valid_o) begin
                //         log2_N_q      <= log2_log2_o;               // Latch result
                //         current_state <= STATE_CALC_LOG_SUM_SQ;
                //         end
                // end

                STATE_CALC_LOG_SUM_SQ: begin
                        log2_valid_i  <= 1'b1;
                        // Convert integer ?c² to fixed-point format
                        log2_data_i   <= {sum_of_squares, {8'd0}};
                        current_state <= STATE_WAIT_LOG_SUM_SQ;
                end

                STATE_WAIT_LOG_SUM_SQ: begin
                        if (log2_valid_o) begin
                        log2_sum_sq_q <= log2_log2_o;
                        current_state <= STATE_DONE;
                        end
                end

                STATE_DONE: begin
                        // Perform final calculation: H? = 2*log?(N) - log?(?c²)
                        entropy_renyi2 <= (LOG2_N_Q << 1) - log2_sum_sq_q;
                        entropy_valid  <= 1'b1;
                        current_state  <= STATE_IDLE;
                end
                
                default: current_state <= STATE_IDLE;
                endcase
        end
end



//==================================================================================
// Module Instantiation
//==================================================================================
    // --- Histogram Module ---
entropy_histogram #(
        .PIXEL_WIDTH                       (8                              ),
        .BLOCK_WIDTH                       (320                            ),
        .BLOCK_HEIGHT                      (240                            ) 
)u_entropy_histogram(
        .clk                               (clk                            ),
        .rst_n                             (rst_n                          ),
        .i_hblank                          (i_hblank                       ),
        .i_vblank                          (i_vblank                       ),
        .i_data                            (i_data                         ),

        .valid_results                     (his_valid_results              ),
        .his_data                          (his_data                       ),
        .his_data_n                        (his_data_n                     )
);
// count_(i)^2  3 pip
mult_gen_square u_mult_gen_square (
        .CLK                               (clk                            ),// input wire CLK
        .B                                 (his_data_n                     ),// input wire [15 : 0] B
        .A                                 (his_data_n                     ),// input wire [15 : 0] A
        .P                                 (mult_gen_square_P              ) // output wire [31 : 0] P
);


    // --- Log2 Module ---
    log2cordic#(
        .INPUT_WIDTH                       (32             ),// e.g., 34
        .INPUT_POINT                       (8                             ),// Using Q.16 for better precision
        .OUTPUT_POINT                      (16                             ),
        .OUTPUT_WIDTH                      (32                             ),// Q16.16 format
        .STAGES                            (LOG2_LATENCY                   ) 
    ) u_log2cordic(
        .clock                             (clk                            ),
        .reset                             (~rst_n                         ),// Assuming active-high reset
        .data_i                            (log2_data_i                    ),
        .valid_i                           (log2_valid_i                   ),
        .log2_o                            (log2_log2_o                    ),
        .valid_o                           (log2_valid_o                   ) 
    );
endmodule