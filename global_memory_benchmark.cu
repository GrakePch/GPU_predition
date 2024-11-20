#include <iostream>
#include <cuda.h>

// Kernel for global memory benchmark
__global__ void benchmarkGlobalMemory(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        #pragma unroll 1000
        for (int i = 0; i < 1000; ++i) {
            data[idx] += 1.0f;  // test global memory access
        }
    }
}

int main() {
    // Hardware parameters (update as needed)
    const float BWg = 652.8e9;  // Global memory bandwidth (bytes/s)
    const float f_gpu = 1.455e9; // GPU clock rate (Hz)
    const int Szgl = 384;       // Global memory transaction size (bytes)
    const int mc = 64;           // Max hidden latency
    const int ILP = 1;          // Instruction-level parallelism
    const int TLP = 64;         // Thread-level parallelism (max warps)

    // Allocate global memory
    const int N = 1 << 20;
    float *data;
    cudaMallocManaged(&data, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        data[i] = 0.0f;
    }

    // Timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel and measure execution time
    cudaEventRecord(start);
    benchmarkGlobalMemory<<<(N + TLP - 1) / TLP, TLP>>>(data, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate base latency (lmi) in seconds
    float lmi = milliseconds / (1000 * N); // Average time per access

    // Calculate pm (penalty for serialization)
    float pm = (float)Szgl / BWg;

    // Calculate memory delay (dmi)
    float dmi;
    if (ILP * TLP <= mc) {
        dmi = lmi / (ILP * TLP);
    } else {
        dmi = lmi / (ILP * TLP * mc) + pm;
    }

    // Convert delay to clock cycles
    float dmi_cycles = dmi * f_gpu;

    // Output results
    std::cout << "Global memory base latency (lmi): " << lmi << " seconds" << std::endl;
    std::cout << "Penalty (pm): " << pm << " seconds" << std::endl;
    std::cout << "Global memory delay (dmi): " << dmi << " seconds" << std::endl;
    std::cout << "Global memory delay in cycles: " << dmi_cycles << " cycles" << std::endl;

    // Free memory
    cudaFree(data);

    return 0;
}
