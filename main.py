import sys
from cfg import cfg

# if len(sys.argv) < 2:
#     print ("Usage: need arguments <filename>")
#     sys.exit(1)
    
# filename = sys.argv[1]


# kernel = cfg(filename)  # get the code block flow from PTX

# kernel.print()

# TODO: Get from Hardware Manual
max_wSM = 64   # Max number of active warps per SM. Hardare limit for kepler+  it is 64
max_tSM = 2048   # max number of threads per SM
max_bSM = 32        # Max number of block per SM

# TODO: Get from Device Query
n_SM =  80     # num of SMs
n_cSM = 64     # num of SPs per SM
Sz_w = 32        # Warp size to schedule a set of threads
Sz_gl = 3072 / 8 # bytes, Num of bytes in a single global mem transaction.
f_gpu = 1.455 * 1e9   # Hz, GPU Max Clock rate

### TODO: Get from user input
# Kernel config
n_b = 1  # number of blocks
n_tb = 2047  # number of threads per block
n_t = n_b * n_tb  # number of threads total

ILP = 1 # assume or instruction are dependent. If independent, ILP could be higher.

TLP = max_wSM   # For simpler model

BW_g = 585.5 * 1e9 # Byte/s, Bandwidth for global load/store transfers. Get from bandwidth_test.cu


d_k = f_gpu ##TODO: Total cycle delay for 1 thread

### Kernel launch overhead
l_overhead = 6.3331e-12 * n_t + 3.7808e-06

### Memory Bottlenecks penalty
b_penalty = 0 ## TODO:

### Simulation algorithm
t_thread = d_k / f_gpu  # delay in seconds for 1 thread
n_ic = n_tb / n_cSM     # number of issue cycles per thread: 
                        # quantifies the extra cycles needed to issue all instructions for all warps in a thread block

t_thread = t_thread * n_ic  

ts_kernel = 0   # Accumulated delay for all blocks

remain_blocks = n_b
while remain_blocks > 0:
    num_blocksRound = 0
    currentSM = 0
    SM_counters = [0] * n_SM
    
    while num_blocksRound < max_bSM * n_SM and remain_blocks > 0:
        SM_counters[currentSM] += 1
        print(SM_counters)
        num_blocksRound += 1
        currentSM = (currentSM + 1) % n_SM  # Round-robin assignment
        remain_blocks -= 1
            
    ts_kernel += max(SM_counters) * t_thread
    
t_kernel = ts_kernel + l_overhead + b_penalty

print(ts_kernel)
    