#include <iostream>
#include <cuda.h>

// Kernel for benchmarking add.f32
__global__ void benchmarkAdd(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        #pragma unroll 1000
        for (int i = 0; i < 1000; ++i) {
            c[idx] = a[idx] + b[idx];  // test for add.f32
        }
    }
}

int main() {
    // Parameters from deviceQuery and manual
    const int max_wSM = 64;   // Max number of active warps per SM
    const int max_tSM = 2048; // Max number of threads per SM
    const int Sz_w = 32;      // Warp size
    const float f_gpu = 1.455e9; // GPU Max Clock rate (Hz)
    const float BW_g = 652.8e9; // Bandwidth for global memory transactions (bytes/s)

    const int N = 1 << 20;  // 1M data elements
    float *a, *b, *c;
    cudaMallocManaged(&a, N * sizeof(float));
    cudaMallocManaged(&b, N * sizeof(float));
    cudaMallocManaged(&c, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Kernel configuration
    const int n_tb = 256; // Number of threads per block
    const int n_b = (N + n_tb - 1) / n_tb;  // Number of blocks
    const int n_t = n_b * n_tb; // Total threads
    const int ILP = 1;    // Instruction-level parallelism we set it to 1
    const int TLP = max_wSM; // Warp-level parallelism and TLP = max_wSM 

    int mc = 64;  // Maximum hidden latency, hardware-specific

    // Timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel and measure execution time
    cudaEventRecord(start);
    benchmarkAdd<<<n_b, n_tb>>>(a, b, c, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    // Compute single instruction base delay in seconds
    float lci = milliseconds / (1000 * N); // Average time per instruction (seconds)
    std::cout << "lci (basic delay per instruction): " << lci << " seconds" << std::endl;

    float Szw = 384; // Total data size we get from query
    float tpc = Szw / BW_g; // Time per transaction in seconds

    // Compute instruction delay (dci) based on formula
    float dci;
    if (ILP * TLP <= mc) {
        dci = lci / (ILP * TLP);
    } else {
        dci = lci / (ILP * TLP * mc) + Szw / tpc;
    }

    // Convert delay to clock cycles
    float dci_cycles = dci * f_gpu;

    // Output results
    std::cout << "Instruction delay (dci): " << dci << " seconds" << std::endl;
    std::cout << "Instruction delay in cycles: " << dci_cycles << " cycles" << std::endl;

    // Verify output data
    for (int i = 0; i < 10; i++) {
        std::cout << "c[" << i << "] = " << c[i] << std::endl;
    }

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
