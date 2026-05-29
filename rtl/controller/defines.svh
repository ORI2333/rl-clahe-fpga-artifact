// ============================================================================
// 固定点/结构参数定义
// ============================================================================
`ifndef __DEFINES_SVH__
`define __DEFINES_SVH__

// 固定点：Q4.12
`define Q_FRAC_BITS         12

// 并行度（时分复用宽度）
`define P_WIDTH             64

// 各层维度
// `define DIM_IN              7
// `define DIM_L1              256
// `define DIM_L2              256
// `define DIM_L3H             128
// `define DIM_OUT             1
// students 
`define DIM_IN              5
`define DIM_L1              128
`define DIM_L2              64
`define DIM_OUT             1


`endif
