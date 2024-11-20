#include <stdio.h>
#include <cuda.h>

// Kernel to perform repeated add.f32 operations
__global__ void addfThroughput(float *a, float *b, float *c, int iterations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float reg_a = a[idx];
    float reg_b = b[idx];
    float reg_c = 0;

    #pragma unroll
    for (int i = 0; i < iterations; i++) {
        reg_c = reg_a + reg_b; // Single-precision floating-point addition
        reg_a = reg_c;         // Chain dependencies to keep operations in-flight
    }

    c[idx] = reg_c; // Store the result to prevent optimization
}

int main() {
    const int threads_per_block = 256; // Number of threads per block
    const int blocks = 80;             // Number of blocks (adjust for your GPU)
    const int iterations = 100000;     // Number of add.f32 operations per thread

    int num_threads = threads_per_block * blocks;

    // Allocate and initialize memory
    float *h_a = (float *)malloc(num_threads * sizeof(float));
    float *h_b = (float *)malloc(num_threads * sizeof(float));
    float *h_c = (float *)malloc(num_threads * sizeof(float));

    for (int i = 0; i < num_threads; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, num_threads * sizeof(float));
    cudaMalloc(&d_b, num_threads * sizeof(float));
    cudaMalloc(&d_c, num_threads * sizeof(float));

    cudaMemcpy(d_a, h_a, num_threads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, num_threads * sizeof(float), cudaMemcpyHostToDevice);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel and measure execution time
    cudaEventRecord(start);
    addfThroughput<<<blocks, threads_per_block>>>(d_a, d_b, d_c, iterations);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate throughput
    long long total_operations = (long long)num_threads * iterations; // Total add.f32 operations
    float giga_ops = (total_operations / (milliseconds / 1000.0f)) / 1e9; // GFLOPS

    printf("Throughput: %.2f GFLOPS\n", giga_ops);

    // Calculate peak warps
    int warp_size = 32; // Typical warp size
    int active_warps = num_threads / warp_size;
    printf("Active Warps: %d\n", active_warps);

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
