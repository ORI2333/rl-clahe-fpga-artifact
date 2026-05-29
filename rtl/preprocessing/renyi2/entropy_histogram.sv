`timescale 1ns / 1ps
module entropy_histogram #(
        parameter          PIXEL_WIDTH      = 8                    ,
        parameter          BLOCK_WIDTH      = 320                  ,
        parameter          BLOCK_HEIGHT     = 240                  
)(
        input  wire                                       clk                 ,
        input  wire                                       rst_n               ,
        input  wire                                       i_hblank            ,
        input  wire                                       i_vblank            ,
        input  wire         [   PIXEL_WIDTH-1: 0]         i_data              ,

        output reg                                        valid_results       ,
        output wire         [   PIXEL_WIDTH-1: 0]         his_data            ,
        output wire         [              15: 0]         his_data_n           
        //output reg          [ PIXEL_WIDTH*2-1: 0]         grayscale_grayscale_Value
);
        localparam         IMG_GRAY         = 256                  ;
        localparam         TOTAL_PIXELS     = BLOCK_WIDTH*BLOCK_HEIGHT;

        localparam         IMAGE_MAX        = $clog2(IMG_GRAY+1) ;
        localparam         TOTAL_PIXEL      = $clog2(TOTAL_PIXELS*IMG_GRAY);

        localparam         IDLE             = 3'b000               ;
        localparam         CUL              = 3'b001               ;
        localparam         GET              = 3'b010               ;
        localparam         CLEAR            = 3'b011               ;

reg            [              11: 0]     row_cnt                ;
reg                                      hblank_dly_1           ;
reg                                      hblank_dly_2           ;

wire                                     hblank_neg_edge        ;

reg            [               2: 0]     state                  ;

// determine pixel
reg                                      pix_same_flag          ;
reg            [   TOTAL_PIXEL-1: 0]     pix_same_cnt           ;
reg            [   TOTAL_PIXEL-1: 0]     pix_same_cnt_dly_1     ;
reg            [   PIXEL_WIDTH-1: 0]     data_dly1              ;
reg            [   PIXEL_WIDTH-1: 0]     data_dly_2             ;

reg            [   TOTAL_PIXEL-1: 0]     cul_data               ;
reg            [     IMAGE_MAX-1: 0]     get_cnt                ;

    // RAM control signals
reg                                      his_wea                ;
reg                                      his_enb                ;
reg            [   PIXEL_WIDTH-1: 0]     his_addra              ;
reg            [   PIXEL_WIDTH-1: 0]     his_addrb              ;
reg            [              15: 0]     his_dina               ;
wire           [              15: 0]     his_doutb              ;

// delay hblank
always @(posedge clk or negedge rst_n) begin                                        
    if(!rst_n) begin                      
        hblank_dly_1 <= 1'b0;
        hblank_dly_2 <= 1'b0;
        data_dly1 <= 'd0;
        data_dly_2 <= 'd0;
    end else begin
        hblank_dly_1 <= i_hblank;
        hblank_dly_2 <= hblank_dly_1;
        data_dly1 <= i_data;
        data_dly_2 <= data_dly1;
    end
end
// capture hblank negedge 
assign hblank_neg_edge = (hblank_dly_1) && (!i_hblank);
 // row counter : source blanking signal
always @(posedge clk or negedge rst_n) begin
    if(!rst_n)
        row_cnt <= 0;
    else if(i_vblank) begin
        if(hblank_neg_edge)
            row_cnt <= row_cnt + 1;
        else if(row_cnt == BLOCK_HEIGHT)
            row_cnt <= 0;
    end else
        row_cnt <= 0;
end

// Determine if the address is the same
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        pix_same_flag <= 1'b0;
    else if (hblank_dly_1) begin
        if (i_data != data_dly1)
        // When the grayscale value of the current pixel point and the previous pixel point is different, 
        // the flag position is 1
            pix_same_flag <= 1'b1; 
        else if ((i_hblank == 1'b0) && (hblank_dly_1 == 1'b1))
            pix_same_flag <= 1'b1;
        else
            pix_same_flag <= 1'b0;
    end else
        pix_same_flag <= 1'b0;
end

// The number of adjacent pixel points 
// with the same grayscale value will start from the second pixel point
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) 
        pix_same_cnt    <= 'd1;
    else if((i_hblank == 1'b1) ) begin
        if(i_data == data_dly1) 
            pix_same_cnt <= pix_same_cnt + 'd1;
            else
            pix_same_cnt <= 'd1;
    end else begin
        pix_same_cnt <= 'd0;
    end
end
// delay pix_same_cnt
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) 
        pix_same_cnt_dly_1 <= 'd0;
    else  begin
        pix_same_cnt_dly_1 <= pix_same_cnt;
    end
end

// cul data
always @(posedge clk or negedge rst_n) begin
    if(!rst_n)
        cul_data <= 'd0;
    else if(state == CUL) begin
        //if(pix_same_flag)
            cul_data <= his_doutb + pix_same_cnt;
    end else begin
        cul_data <= 'd0;
    end
end


// write port (A)
always @(*) begin
    if(state == CLEAR) begin
        his_wea <= 1'b1;
        his_addra <= get_cnt;
        his_dina <= 'd0;
    end else if(state == CUL) begin
        his_wea <= pix_same_flag;
        his_addra <= data_dly_2;
        his_dina <= cul_data;
    end else begin
        his_wea <= 0;
        his_addra <= 0;
        his_dina <= 0;
    end
end

// read port (B)
always @(*) begin
    if(state == CUL) begin
        his_enb <= i_hblank || hblank_dly_1;
        his_addrb <= i_data;
    end else if(state == GET) begin
        his_enb <= 1'b1;
        his_addrb <= get_cnt;
    end else begin
        his_enb <= 0;
        his_addrb <= 0;
    end
end

// read data signal
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        get_cnt <= 'd0;
    else if(state == GET)
        if (get_cnt < IMG_GRAY - 1)
            get_cnt <= get_cnt + 1;
       
    else if (state == CLEAR) begin
        if (get_cnt > 'd0) begin
        get_cnt <=  get_cnt - 'd1;
        end
    end 
end

// FSM
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        state   <= IDLE;
    end else begin
        case(state)
            IDLE: begin
                if(i_hblank == 1'b1 && i_vblank == 1'b1)
                    state   <= CUL; 
                else  
                    state   <= IDLE;
            end
            CUL: begin
                if((row_cnt == BLOCK_HEIGHT) && (i_hblank == 1'b0))
                    state   <= GET;
                else
                    state  <= CUL;
            end
            GET: begin
                if(get_cnt == IMG_GRAY - 1)
                    state   <= CLEAR;
                else
                    state   <= GET;
            end
            CLEAR: begin
                if(get_cnt == 'd0) begin
                    state   <= IDLE;
                end else begin
                    state   <= CLEAR;
                end
            end
            default: state   <= IDLE;   
        endcase  
    end
end

// output 
assign   valid_results    = (state == GET)       ;
assign   his_data         = (state == GET) ? get_cnt : 'd0;
assign   his_data_n       = (state == GET) ? his_doutb : 'd0;

    // always @(posedge clk) begin
    //     grayscale_grayscale_Value <= {get_cnt, his_doutb};
    // end

    blk_mem_gen_his_get u_blk_mem_gen_his_get (
        .clka                              (clk                            ),// input wire clka
        .ena                               (1'b1                           ),// input wire ena
        .wea                               (his_wea                        ),// input wire [0 : 0] wea
        .addra                             (his_addra                      ),// input wire [7 : 0] addra
        .dina                              (his_dina                       ),// input wire [15 : 0] dina
        .clkb                              (clk                            ),// input wire clkb
        .enb                               (his_enb                        ),// input wire enb
        .addrb                             (his_addrb                      ),// input wire [7 : 0] addrb
        .doutb                             (his_doutb                      ) // output wire [15 : 0] doutb
);

endmodule
