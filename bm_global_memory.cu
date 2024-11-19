#include <stdio.h>
#include <cuda.h>

// Kernel for measuring global memory latency
__global__ void globalMemoryLatency(int *d_data, int iterations, int *result) {
    int idx = threadIdx.x; 
    int value = idx;

    for (int i = 0; i < iterations; i++) {
        value = d_data[value];
    }

    result[idx] = value; // Prevent compiler optimization
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("usage:  gpu_frequency in Ghz\n");
        exit(1);
    }

    float f = atof(argv[1]);
    printf("GPU frequency %f Ghz.\n", f);

    const int iterations = 1e7; // Number of memory accesses
    const int array_size = 1024;    // Size of the array for pointer chasing
    const int thread_count = 1;     // Single thread for accurate measurement

    int *h_data, *d_data, *d_result;
    cudaEvent_t start, stop;

    // Allocate host memory
    h_data = (int *)malloc(array_size * sizeof(int));
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return -1;
    }

    // Initialize data for pointer chasing
    for (int i = 0; i < array_size; i++) {
        h_data[i] = (i + 1) % array_size; // Circular indexing
    }

    // Allocate device memory
    cudaMalloc(&d_data, array_size * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_data, h_data, array_size * sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    // Launch the kernel
    globalMemoryLatency<<<1, thread_count>>>(d_data, iterations, d_result);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float latency = milliseconds / iterations;
    printf("Global Memory Latency: %f ms\n", latency);
    float cycles = latency / 1000 * f * 1e9;
    printf("Cycle: %f\n", cycles);

    // Free resources
    free(h_data);
    cudaFree(d_data);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
