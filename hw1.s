.data
    n: .word 11
.text
.globl __start

FUNCTION:
    # Todo: Define your own function in HW1
funT:
    # allocate memory for x1(comeback pointer) and a0(n)
    addi sp, sp, -8
    sw x1, 4(sp)
    sw a0, 0(sp)
    # test what segment n belongs to
    addi x5, a0, -10
    bge  x5, x0, L3
    addi x5, a0, -1
    bge x5, x0, L2
    # if reach here, n<=0
    addi a0, x0, 7
    # pop stack and go back to where it is called
    addi sp, sp, 8
    add t0 x0 a0       # place output to t0
    jalr x0, 0(x1)


L2: addi a0, a0, -1      # n = n - 1
    jal x1, funT         # call funT(n-1)
    addi x6, a0, 0       # store the result of funT(n-1) to x6
    lw a0, 0(sp)         # load back n(optional)
    lw x1, 4(sp)         # load back comeback pointer
    addi sp, sp, 8       # pop stack
    add a0, x6, x6       # output 2*funT(n-1)
    jalr x0, 0(x1)
    
L3: # x9 = 0.875n-137
    addi x7, x0, 7
    mul x9, x7, a0
    srli x9, x9, 3
    addi x9, x9, -137
    
    # call funT(3/4n)
    addi x7, x0, 3
    mul a0, a0, x7
    srli a0, a0, 2
    jal x1, funT
    
    # x7 = 2*funT(3/4n)
    addi x7, a0, 0
    add x7, x7, x7
    
    lw a0, 0(sp) 
    lw x1, 4(sp)     # load back to where funT is first called
    addi sp, sp, 8
    add a0, x7, x9   # T(n) = 2T(3n/4) + 0.875n - 137
    jalr x0, 0(x1)

# Do NOT modify this part!!!
__start:
    la   t0, n
    lw   x10, 0(t0)
    jal  x1,FUNCTION
    la   t0, n
    sw   x10, 4(t0)
    addi a0,x0,10
    ecall