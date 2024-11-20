import sys
import math
from cfg import cfg

if len(sys.argv) < 4:
    print ("Usage: python main.py <ptx_file_path> <total_threads> <loop_iter>")
    sys.exit(1)
    
filename = sys.argv[1]


kernel = cfg(filename)  # get the code block flow from PTX

# kernel.print()

# Get from Hardware Manual
max_wSM = 64   # Max number of active warps per SM. Hardare limit for kepler+  it is 64
max_tSM = 2048   # max number of threads per SM
max_bSM = 32        # Max number of block per SM

# Get from Device Query
n_SM =  80     # num of SMs
n_cSM = 64     # num of SPs per SM
Sz_w = 32        # Warp size to schedule a set of threads
Sz_gl = 3072 / 8 # bytes, Num of bytes in a single global mem transaction.
f_gpu = 1.455 * 1e9   # Hz, GPU Max Clock rate

# Kernel config, Get from user input
n_t = int(sys.argv[2])  # number of threads total
custom_loop_iterations = int(sys.argv[3])  # number of loops' iteration, cannot be known from ptx
c_coalescing_factor = 1     # coalescing factor =1 means fully coalescing. >1 means less coalescing.

n_tb = 256    # number of threads per block
n_b = n_t / n_tb  # number of blocks

ILP = 1 # assume or instruction are dependent. If independent, ILP could be higher.

TLP = max_wSM   # For simpler model
# TLP = 1   # For simpler model

BW_g = 585.5 * 1e9 # Byte/s, Bandwidth for global load/store transfers. Get from bandwidth_test.cu

### Microbenchmark Temp Test Result: [cycles, throughput, peakwarp]
dictBM = {
    "add": [4, None, 64],
    "sub": [4, None, 64],
    "mul": [4, None, 64],
    "mad": [5, None, 64],
    "fma": [4, None, 64],
    "div": [481, None, 64],
    "mov": [4, None, 64],
    "and": [2, None, 64],
    "or": [2, None, 64],
    "setp": [4, None, 64],
    "cvta": [10, None, 64],
}

# [cycles, peakwarp]
dictBM_mem = {
    "ld.global": [437, 64],
    "st.global": [446, 64],
    "st.volatile.global": [446, 64]
}


### Delay calculation for 1 thread
def instructionCompDelay(l, t, m):
    d = l / TLP
    if (TLP > m):
        penalty = Sz_w / t
        d = d / m + penalty
    return d

def instructionMemDelay(l, m):
    d = l / TLP
    if (TLP > m):
        penalty = Sz_gl * c_coalescing_factor / BW_g * f_gpu
        d += penalty
    return d


def instructionD(inst):
    if inst.startswith("ld") or inst.startswith("st"):
        try:
            l, m = dictBM_mem[inst[:-4]]
            return instructionMemDelay(l, m)
        except:
            return 0
    else:
        try:
            l, t, m = dictBM[inst.split(".")[0]]
            return instructionCompDelay(l, t, m)
        except:
            print(inst, "not found!")
            return 0
        

def delayCalculation(kernel):
    isFirstLoop = True
    d_kernal = 0
    for basicBlock in kernel.listBasicBlock:
        d_b = 0
        for i in basicBlock.dictInstMem:
            num_i = basicBlock.dictInstMem[i]
            d_b += instructionD(i) * num_i
        for i in basicBlock.dictInstComp:
            num_i = basicBlock.dictInstComp[i]
            d_b += instructionD(i) * num_i
        
        if basicBlock.isLoop and isFirstLoop:
            d_b *= custom_loop_iterations
            isFirstLoop = False
            
        d_kernal += d_b
    return d_kernal
            
        
# Total cycle delay for 1 thread
d_k = delayCalculation(kernel)

print("1 Thread delay:", d_k, "cycles")

### Kernel launch overhead
l_overhead = 6.3331e-12 * n_t + 3.7808e-06

### Memory Bottlenecks penalty
b_penalty = 0 

### Simulation algorithm
# t_thread = d_k / f_gpu  # delay in seconds for 1 thread
# n_ic = n_tb / n_cSM     # number of issue cycles per thread: 
#                         # quantifies the extra cycles needed to issue all instructions for all warps in a thread block

# t_thread = t_thread * n_ic  

# ts_kernel = 0   # Accumulated delay for all blocks

# remain_blocks = n_b
# while remain_blocks > 0:
#     num_blocksRound = 0
#     currentSM = 0
#     SM_counters = [0] * n_SM
    
#     while num_blocksRound < max_bSM * n_SM and remain_blocks > 0:
#         SM_counters[currentSM] += 1
#         num_blocksRound += 1
#         currentSM = (currentSM + 1) % n_SM  # Round-robin assignment
#         remain_blocks -= 1
            
#     ts_kernel += max(SM_counters) * t_thread
    
# t_kernel = ts_kernel + l_overhead + b_penalty

# print(ts_kernel)



### Below are another implementation of Algorithm 2. 根据我自己的理解来写

max_reg_per_SM = 65536   # Hardware limit

def registers_used_per_thread(kernelPTX):
    reg_usage = kernelPTX.dictReg
    num_reg = 0
    for k in reg_usage:
        if k == "pred": continue
        bit_size_str = k[1:]
        bit_size = 0
        try:
            bit_size = int(bit_size_str)
        except ValueError:
            print("Unknown size reg:", k)
            bit_size = 0
        num_reg_this_type = reg_usage[k]
        num_reg += math.ceil(bit_size * num_reg_this_type / 32)
    return num_reg
            
    
    
def Custom_Sim_Algo():
    time_thread = d_k / f_gpu                     # Time to run one thread
    num_ic = math.ceil(n_tb / n_cSM)              # Num of "small wave" of parallel threads in one block
    time_block = time_thread * num_ic             # Time to run one block
    blocks_to_exe = n_b                           # Time to run one block
    num_block_limit_by_thread = max_tSM // n_tb   # block num limited by max thread in one SM
    
    ### 待删除评论：每个SM能分配多少Block也受到register总数的限制，paper里似乎没有详细处理
    num_reg_per_thread = registers_used_per_thread(kernel)        # registers used per thread
    num_reg_per_block = num_reg_per_thread * n_tb                 # registers used per block
    num_block_limit_by_reg = max_reg_per_SM // num_reg_per_block  # block num limited by registers in one SM
    
    max_block_per_sm = min(
        num_block_limit_by_thread, 
        max_bSM, 
        num_block_limit_by_reg)    # max blocks per SM in ONE WAVE.
    
    # Assume RoundRobin scheduling
    max_block_per_wave = max_block_per_sm * n_SM                  # max blocks total in ONE WAVE. 
    num_waves = math.ceil(blocks_to_exe / max_block_per_wave)     # Num of WAVE to run all blocks
    ### TODO: 待删除评论：如果round robin的话根本没必要用SMcounter[currentSM]一个一个模拟过去吧。
    
    totalTime = num_waves * time_block
    return totalTime + l_overhead + b_penalty

finalT = Custom_Sim_Algo()
print("Predicted: {:.6f} s".format(finalT))
    
    
    