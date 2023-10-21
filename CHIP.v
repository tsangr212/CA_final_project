module CHIP(clk,
            rst_n,
            // For mem_D
            mem_wen_D,
            mem_addr_D,
            mem_wdata_D,
            mem_rdata_D,
            // For mem_I
            mem_addr_I,
            mem_rdata_I);

input         clk, rst_n ;
// For mem_D
output        mem_wen_D  ;  // 0: read from data/stack memory, 1: write to ...
output [31:0] mem_addr_D ;  // address of memory
output [31:0] mem_wdata_D;  // data written to memory
input  [31:0] mem_rdata_D;  // data read from memory
// For mem_I
output [31:0] mem_addr_I ;  // address of I memory
input  [31:0] mem_rdata_I;  // instruction read from I memory

//---------------------------------------//
// Do not modify this part!!!            //
// Exception: You may change wire to reg //
reg    [31:0] PC          ;              //
reg    [31:0] PC_nxt      ;              //
wire          regWrite    ;              //
wire   [ 4:0] rs1, rs2, rd;              //
wire   [31:0] rs1_data    ;              //
wire   [31:0] rs2_data    ;              //
wire   [31:0] rd_data     ;              //
//---------------------------------------//

// Todo: other wire/reg
// wire [31:0] data1;
// wire [31:0] data2;
//---------------------------------------//
// Do not modify this part!!!            //
reg_file reg0(                           //
             .clk(clk),                           //
             .rst_n(rst_n),                       //
             .wen(regWrite),                      //
             .a1(rs1),                            //
             .a2(rs2),                            //
             .aw(rd),                             //
             .d(rd_data),                         //
             .q1(rs1_data),                       //
             .q2(rs2_data));                      //
//---------------------------------------//

// Todo: any combinational/sequential circuit
/* ------------------------------------------------------------------ */
/*                               Control                              */
/* ------------------------------------------------------------------ */
wire branch;
wire jal;
wire jalr;
wire is_AUIPC;
wire [3:0] ALU_control;
wire ALU_src;
wire [31:0] imm;



Control Control0(
            .mem_rdata_I(mem_rdata_I),
            .branch(branch),
            .jal(jal),
            .jalr(jalr),
            .mem_wen_D(mem_wen_D),
            .mem_to_reg(mem_to_reg),
            .ALU_control(ALU_control),
            .ALU_src(ALU_src),
            .regWrite(regWrite),
            .imm(imm),
            .is_AUIPC(is_AUIPC));
/* ------------------------------------------------------------------ */
/*                               ALU                                  */
/* ------------------------------------------------------------------ */
wire [31:0] ALU_data1;
wire [31:0] ALU_data2;
wire [31:0] ALU_result;
wire ALU_zero;

ALU ALU0(.data1(ALU_data1),
         .data2(ALU_data2),
         .ALU_control(ALU_control),              // ALU_control is ALU's control signal, ALU_Control is ALU_Control unit's output
         .ALU_result(ALU_result),
         .ALU_zero(ALU_zero));

/* --------------------------------------------------------------------- */
/*                               Multiplier                              */
/* --------------------------------------------------------------------- */
wire mulready;
wire [63:0] mul_result;
reg mulmode;
wire PC_pause;
reg [1:0] mode ;


assign PC_pause = (mulmode & !mulready)? 1'd1 : 1'd0;

always @(*)
begin
    if(ALU_control == 4'b0100) //mul
    begin
        mulmode = 1;
        mode = 2'd0;
    end
    else if(ALU_control == 4'b1100) //div
    begin
        mulmode = 1;
        mode = 2'd1;
    end
    else
    begin
        mulmode = 0;
        mode = 2'd0;
    end
end


mulDiv mulDiv0(.clk(clk),
               .rst_n(rst_n),
               .valid(mulmode),
               .ready(mulready),
               .mode(mode),
               .in_A(rs1_data),
               .in_B(rs2_data),
               .out(mul_result));
/* ------------------------------------------------------------------ */
/*                               PC_IMM ADDER                         */
/* ------------------------------------------------------------------ */

//wire [31:0] PC_addr;
wire [31:0] sum_addr;
wire [31:0]base_addr;
assign base_addr = (jalr)? rs1_data : PC;

PC_IMM_ADDER PC_IMM_ADDER0(.PC_addr(base_addr),
                           .imm(imm),
                           .sum_addr(sum_addr));


/* -------------------------------------------------------------------------- */
/*                           Assign wire connection                           */
/* -------------------------------------------------------------------------- */

always @(*)
begin
    PC_nxt = (PC_pause)? PC :
           ((branch & ALU_zero) | jal | jalr)? sum_addr : PC + 4;
end

assign mem_addr_I = PC;
assign mem_wdata_D = rs2_data;
assign mem_addr_D = ALU_result;
assign rs1 = mem_rdata_I[19:15];
assign rs2 = mem_rdata_I[24:20];
assign rd  = mem_rdata_I[11:7];
assign rd_data = (rd == 5'd0)? 32'd0 :
       (mem_to_reg)? mem_rdata_D :
       (is_AUIPC)? sum_addr :
       (jal | jalr)? PC+4 :
       (mulmode)? mul_result[31:0] : ALU_result;

assign ALU_data1 = (is_AUIPC | jal | jalr)? PC : rs1_data;
assign ALU_data2 = (ALU_src)? imm : rs2_data;


/* -------------------------------------------------------------------------- */
/*                           Sequential circuit                               */
/* -------------------------------------------------------------------------- */

always @(posedge clk or negedge rst_n)
begin
    if (!rst_n)
    begin
        PC <= 32'h00010000; // Do not modify this value!!!
    end
    else
    begin
        PC <= PC_nxt;
    end
end

endmodule


    /* -------------------------------------------------------------------------- */
    /*                              reg_file(module)                              */
    /* -------------------------------------------------------------------------- */

    module reg_file(clk, rst_n, wen, a1, a2, aw, d, q1, q2);

parameter BITS = 32;
parameter word_depth = 32;
parameter addr_width = 5; // 2^addr_width >= word_depth

input clk, rst_n, wen; // wen: 0:read | 1:write
input [BITS-1:0] d;
input [addr_width-1:0] a1, a2, aw;

output [BITS-1:0] q1, q2;

reg [BITS-1:0] mem [0:word_depth-1];
reg [BITS-1:0] mem_nxt [0:word_depth-1];

integer i;

assign q1 = mem[a1];
assign q2 = mem[a2];

always @(*)
begin
    for (i=0; i<word_depth; i=i+1)
        mem_nxt[i] = (wen && (aw == i)) ? d : mem[i];
end

always @(posedge clk or negedge rst_n)
begin
    if (!rst_n)
    begin
        mem[0] <= 0;
        for (i=1; i<word_depth; i=i+1)
        begin
            case(i)
                32'd2:
                    mem[i] <= 32'hbffffff0;
                32'd3:
                    mem[i] <= 32'h10008000;
                default:
                    mem[i] <= 32'h0;
            endcase
        end
    end
    else
    begin
        mem[0] <= 0;
        for (i=1; i<word_depth; i=i+1)
            mem[i] <= mem_nxt[i];
    end
end
endmodule
    /* -------------------------------------------------------------------------- */
    /*                             Control(Module)                                */
    /* -------------------------------------------------------------------------- */

    module Control(
        mem_rdata_I,
        branch,
        jal,
        jalr,
        mem_wen_D,
        mem_to_reg,
        ALU_control,
        ALU_src,
        regWrite,
        imm,
        is_AUIPC);

input  [31:0]   mem_rdata_I;
output          branch;//
output          jal;
output          jalr;
output          mem_wen_D;//
output          mem_to_reg;//
output [3:0]    ALU_control;//
output          ALU_src;//
output          regWrite;//
output [31:0]   imm;//
output          is_AUIPC;

wire [6:0] opcode;
wire [2:0] funct3;
wire [6:0] funct7;

wire [31:0] i_imm;
wire [31:0] s_imm;
wire [31:0] b_imm;
wire [31:0] u_imm;
wire [31:0] j_imm;
wire [31:0] shift_imm;

assign opcode = mem_rdata_I[6:0];
assign funct3 = mem_rdata_I[14:12];
assign funct7 = mem_rdata_I[31:25];

assign branch = (opcode == 7'b1100011) ? 1'b1:1'b0;

assign i_imm = {{20{mem_rdata_I[31]}}, mem_rdata_I[31:20]};
assign s_imm = {{20{mem_rdata_I[31]}}, mem_rdata_I[31:25], mem_rdata_I[11:7]};
assign b_imm = {{20{mem_rdata_I[31]}}, mem_rdata_I[7], mem_rdata_I[30:25], mem_rdata_I[11:8], 1'b0};
assign u_imm = {mem_rdata_I[31:12], 12'b0};
assign j_imm = {{12{mem_rdata_I[31]}}, mem_rdata_I[19:12], mem_rdata_I[20], mem_rdata_I[30:21], 1'b0};
assign shift_imm = {27'b0, mem_rdata_I[24:20]};

localparam [6:0] R_type = 7'b0110011,
           I_type = 7'b0010011,
           AUIPC  = 7'b0010111,
           LOAD   = 7'b0000011,
           STORE  = 7'b0100011,
           BRANCH = 7'b1100011,
           JAL    = 7'b1101111,
           JALR   = 7'b1100111;

assign mem_wen_D   = (opcode == STORE);
assign mem_to_reg  = (opcode == LOAD);
assign ALU_control = (opcode == AUIPC) ? 4'b0000: //aupic > add
       (opcode == LOAD)  ? 4'b0000: //lw    > add
       (opcode == STORE) ? 4'b0000: //sw    > add
       (opcode == JAL)   ? 4'b0010: //jal   > jump
       (opcode == JALR)  ? 4'b0010: //jalr  > jump
       (opcode == BRANCH & funct3 == 3'b000) ? 4'b0111: //beq > beq
       (opcode == BRANCH & funct3 == 3'b101) ? 4'b1000: //bge > bge
       (opcode == R_type & funct3 == 3'b000 & funct7 == 7'b0000000) ? 4'b0000: //add > add
       (opcode == R_type & funct3 == 3'b000 & funct7 == 7'b0000001) ? 4'b0100: //mul > mul
       (opcode == R_type & funct3 == 3'b100 & funct7 == 7'b0000001) ? 4'b1100: //div > div
       (opcode == R_type & funct3 == 3'b000 & funct7 == 7'b0100000) ? 4'b0001: //sub > sub
       (opcode == R_type & funct3 == 3'b100 & funct7 == 7'b0000000) ? 4'b0011: //xor > xor
       (opcode == I_type & funct3 == 3'b000) ? 4'b0000: //addi > add
       (opcode == I_type & funct3 == 3'b010) ? 4'b1001: //slti > slti
       (opcode == I_type & funct3 == 3'b001 & funct7 == 7'b0000000) ? 4'b0110: //slli > slli
       (opcode == I_type & funct3 == 3'b101 & funct7 == 7'b0100000) ? 4'b0101: //srai > srai
       (opcode == I_type & funct3 == 3'b101 & funct7 == 7'b0000000) ? 4'b1101: //srli > div
       4'b0000;
assign ALU_src  = (opcode == AUIPC)  ? 1'b1:
       (opcode == LOAD)   ? 1'b1:
       (opcode == STORE)  ? 1'b1:
       (opcode == JAL)    ? 1'b1:
       (opcode == JALR)   ? 1'b1:
       (opcode == BRANCH) ? 1'b0:
       (opcode == R_type) ? 1'b0:
       (opcode == I_type) ? 1'b1:
       1'b0;
assign regWrite = (opcode == AUIPC)  ? 1'b1:
       (opcode == LOAD)   ? 1'b1:
       (opcode == STORE)  ? 1'b0:
       (opcode == JAL)    ? 1'b1:
       (opcode == JALR)   ? 1'b1:
       (opcode == BRANCH) ? 1'b0:
       (opcode == R_type) ? 1'b1:
       (opcode == I_type) ? 1'b1:
       1'b0;
assign imm = (opcode == I_type & funct3 == 3'b101 & funct7 == 7'b0100000) ? shift_imm:
       (opcode == I_type & funct3 == 3'b101 & funct7 == 7'b0000000) ? shift_imm:
       (opcode == AUIPC)  ? u_imm:
       (opcode == LOAD)   ? i_imm:
       (opcode == STORE)  ? s_imm:
       (opcode == JAL)    ? j_imm:
       (opcode == JALR)   ? i_imm:
       (opcode == BRANCH) ? b_imm:
       (opcode == I_type) ? i_imm:
       32'b0;
assign jal     = (opcode == JAL)  ;
assign jalr    = (opcode == JALR) ;
assign is_AUIPC = (opcode == AUIPC);
endmodule
    /* -------------------------------------------------------------------------- */
    /*                             ALU(Module)                                    */
    /* -------------------------------------------------------------------------- */

    module ALU(
        input [31:0] data1,
        input [31:0] data2,
        input [3:0] ALU_control,        // indicating which one in total 10 operations
        output [31:0] ALU_result,
        output ALU_zero
    );
localparam[3:0] ALU_ADD  = 4'b0000,
          ALU_SUB  = 4'b0001,
          ALU_JUMP = 4'b0010,
          ALU_XOR  = 4'b0011,
          //ALU_MUL  = 4'b0100,
          ALU_SRLI = 4'b1101,
          ALU_SRAI = 4'b0101,
          ALU_SLLI = 4'b0110,
          ALU_BEQ  = 4'b0111,
          ALU_BGE  = 4'b1000,
          ALU_SLTI = 4'b1001;

assign ALU_result = (ALU_control==ALU_ADD) ? data1 + data2 :
       (ALU_control==ALU_SUB) ? data1 - data2 :
       (ALU_control==ALU_JUMP)? data1 + 4 :   //data1 = PC
       (ALU_control==ALU_XOR)? data1^data2 :
       (ALU_control==ALU_SRLI)? data1 >>> data2 :
       //(ALU_control==ALU_MUL)?
       (ALU_control==ALU_SRAI)? data1 >> data2 :
       (ALU_control==ALU_SLLI)? data1 <<< data2 :
       (ALU_control==ALU_SLTI)? ($signed(data1) < $signed(data2)? 1:0) :
       32'd0;

assign ALU_zero =   (ALU_control==ALU_BEQ)? (data1==data2) :
                    (ALU_control==ALU_BGE)? ($signed(data1) >= $signed(data2)):1'd0;

endmodule

    /* -------------------------------------------------------------------------- */
    /*                             PC_IMM ADDER(Module)                           */
    /* -------------------------------------------------------------------------- */
    module PC_IMM_ADDER(
        input [31:0] PC_addr,
        input [31:0] imm,
        output [31:0] sum_addr
    );
assign sum_addr = PC_addr + imm;
endmodule

    /* -------------------------------------------------------------------------- */
    /*                             Multiplier(Module)                             */
    /* -------------------------------------------------------------------------- */

    module mulDiv(
        clk,
        rst_n,
        valid,
        ready,
        mode,
        in_A,
        in_B,
        out
    );

// Definition of ports
input         clk, rst_n;
input         valid;
input  [1:0]  mode; // mode: 0: mulu, 1: divu, 2: and, 3: avg
output        ready;
input  [31:0] in_A, in_B;
output [63:0] out;

// Definition of states
parameter IDLE = 3'd0;
parameter MUL  = 3'd1;
parameter DIV  = 3'd2;
parameter AND = 3'd3;
parameter AVG = 3'd4;
parameter OUT  = 3'd5;
integer  i;

// Todo: Wire and reg if needed
reg  [ 2:0] state, state_nxt;
reg  [ 4:0] counter, counter_nxt;
reg  [63:0] shreg, shreg_nxt;
reg  [31:0] alu_in, alu_in_nxt;
reg  [32:0] alu_out;

// Todo: Instatiate any primitives if needed

// Todo 5: Wire assignments
assign ready = (state==OUT)? 1 : 0;
assign out = (state==OUT)? shreg : 0;

// Combinational always block
// Todo 1: Next-state logic of state machine
always @(*)
begin
    case(state)
        IDLE:
        begin
            if(valid == 0)
                state_nxt = IDLE;
            else if(mode == 0)
                state_nxt = MUL;
            else if(mode == 1)
                state_nxt = DIV;
            else if(mode == 2)
                state_nxt = AND;
            else
                state_nxt = AVG;

        end
        MUL :
        begin
            state_nxt = (counter==31)? OUT : MUL;
        end
        DIV :
        begin
            state_nxt = (counter==31)? OUT : DIV;
        end
        AND :
        begin
            state_nxt = OUT;
        end
        AVG :
        begin
            state_nxt = OUT;
        end
        OUT :
            state_nxt = IDLE;
        default :
            state_nxt = IDLE;
    endcase
end
// Todo 2: Counter
always @(*)
begin
    counter_nxt = (state==MUL || state == DIV)? (counter+1) : counter;
end

// ALU input
always @(*)
begin
    case(state)
        IDLE:
        begin
            if (valid)
                alu_in_nxt = in_B;
            else
                alu_in_nxt = 0;
        end
        OUT :
            alu_in_nxt = 0;
        default:
            alu_in_nxt = alu_in;
    endcase
end

// Todo 3: ALU output
always @(*)
begin
    case(state)
        IDLE:
            alu_out = 0;

        MUL:
            alu_out = {1'b0, alu_in} + {1'b0, shreg[63:32]};

        DIV:
            alu_out = {1'b0, shreg[62:31]} - {1'b0, alu_in};

        AND:
        begin
            for(i=0;i<32;i=i+1)
            begin
                alu_out[i] = shreg[i]&alu_in[i];
            end
            alu_out[32] = 1'b0;
        end

        AVG:
        begin
            alu_out = {1'b0, shreg[31:0]} + {1'b0, alu_in[31:0]};
        end

        default:
            alu_out = {33{1'b1}};
    endcase
end

// Todo 4: Shift register
always @(*)
begin
    case(state)
        IDLE:
        begin
            if(valid)
                shreg_nxt = {{32{1'b0}}, in_A};
            else
                shreg_nxt = 0;
        end

        MUL:
        begin
            if(shreg[0]==1)
            begin
                shreg_nxt = {alu_out, shreg[31:1]};
            end
            else
            begin
                shreg_nxt = {1'b0, shreg[63:1]};
            end
        end

        DIV:
        begin
            if(shreg[62:31] > alu_in)
            begin
                shreg_nxt = {alu_out[31:0], shreg[30:0], 1'b1};
            end
            else
            begin
                shreg_nxt = ({shreg[62:0], 1'b0});
            end
        end

        AND:
        begin
            shreg_nxt = {{32{1'b0}}, alu_out};
        end

        AVG:
        begin
            shreg_nxt = {{32{1'b0}}, alu_out[32:1]};
        end
        default:
            shreg_nxt = shreg;
    endcase
end

// Todo: Sequential always block
always @(posedge clk or negedge rst_n)
begin
    if (!rst_n)
    begin
        state <= IDLE;
        counter <= 0;
        shreg <= 0;
        alu_in <= 0;
    end
    else
    begin
        state <= state_nxt;
        counter <= counter_nxt;
        shreg <= shreg_nxt;
        alu_in <= alu_in_nxt;
    end
end

endmodule

