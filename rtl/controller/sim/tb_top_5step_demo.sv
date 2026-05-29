`timescale 1ns/1ps
`include "defines.svh"

module tb_top_5step_demo;

    logic clk=0   , 
    rst_n=0       , 
    start_pulse=0 ;
    always #5 clk = ~clk; // 100MHz

    // 简单构造 obs7（已归一化后的 Q4.12，实际工程请用 obs_normalizer.svh 在外层变换）
    logic signed [15:0] obs5 [0:`DIM_IN-1];

    wire                                     final_valid                      ;
    wire    signed [              15: 0]     final_cl_q12                     ;

    top_5step_demo dut(
      .clk                               (clk                            ),
      .rst_n                             (rst_n                          ),
      .start_pulse                       (start_pulse                    ),
      .obs_in                            (obs5                           ),
      .final_valid                       (final_valid                    ),
      .final_cl_q12                      (final_cl_q12                   ) 
    );

    initial begin
  $display("[%0t] TB start", $time);
  // 建议复位保持>=5拍，start_pulse 在复位释放后至少 1 拍再拉
end

// 观测 L2 关键信号
always @(posedge clk) begin
  if (dut.u_ctrl.u_mlp.u_layer2.l2_we)
    $display("[%0t] L2 write: addr=%0d data=%0d",
      $time, dut.u_ctrl.u_mlp.u_layer2.l2_waddr, dut.u_ctrl.u_mlp.u_layer2.l2_wdata);
end

// 观测 L2 的 MAC 完成点
always @(posedge clk) begin
  if (&dut.u_ctrl.u_mlp.u_layer2.mac_done_vec)
    $display("[%0t] L2 all 64 MAC done", $time);
end

    initial begin
        // 复位
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(10) @(posedge clk);

        // 构造 7 维：mean/var/entropy/cl/step/brisque/niqe （这里仅演示）
        // obs7[0] = 16'sd1024 ;                                        // 0.25
        // obs7[1] = 16'sd512  ;                                         // 0.125
        // obs7[2] = 16'sd2048 ;                                        // 0.5
        // obs7[3] = 16'sd3072 ;                                        // 当前 cl（0.75）
        // obs7[4] = 16'sd0    ;                                           // step（由 ctrl 内部递增可扩展，这里暂固定）
        // obs7[5] = 16'sd1024 ;                                        // brisque
        // obs7[6] = 16'sd1024 ;                                        // niqe


        obs5[0] = 16'sd0 - 16'sd7452;   // 0.25  (等价于 -16'sd7452，规避非ASCII减号)
        obs5[1] = 16'sd892;             // 0.125
        obs5[2] = 16'sd0 - 16'sd2307;   // 0.5   (等价于 -16'sd2307)
        obs5[3] = 16'sd963;             // 当前 cl（0.75）
        obs5[4] = 16'sd0 - 16'sd5792;   // step  (等价于 -16'sd5792)
        //obs7[5] = 16'sd2292;            // brisque
        //obs7[6] = 16'sd3641;            // niqe

        // 触发
        @(posedge clk); start_pulse = 1;
        @(posedge clk); start_pulse = 0;

        // 等待完成
        wait(final_valid);
        $display("[TB] final_cl_q12 = %0d (Q4.12)", final_cl_q12);
        #100;
        $finish;
    end
endmodule
