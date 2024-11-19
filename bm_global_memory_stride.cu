#include <stdio.h>
#include <cuda.h>

__global__ void strideTestKernel(int *data, int stride, int iterations, int *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int value = idx;

    // Pointer chasing with stride
    for (int i = 0; i < iterations; i++) {
        value = data[(value + stride) % (gridDim.x * blockDim.x)];
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

    const int array_size = 1 << 20; // Large enough to avoid cache effects
    const int iterations = 1000000; // Ensure measurable time
    const int threads_per_block = 256;
    const int blocks = array_size / threads_per_block;

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
        h_data[i] = i;
    }

    // Allocate device memory
    cudaMalloc(&d_data, array_size * sizeof(int));
    cudaMalloc(&d_result, array_size * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_data, h_data, array_size * sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Stride Test Results:\n");
    printf("Stride, Latency (ns)\n");

    // Test for different strides
    for (int stride = 1; stride <= 1024; stride *= 2) {
        // Record start time
        cudaEventRecord(start);

        // Launch the kernel
        strideTestKernel<<<blocks, threads_per_block>>>(d_data, stride, iterations, d_result);

        // Record stop time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calculate elapsed time
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Convert to latency in nanoseconds
        float latency_ns = (milliseconds / iterations) * 1e6;
        printf("%d, %.2f\n", stride, latency_ns);
        float cycles = latency_ns / 1e9 * f * 1e9;
        printf("Cycle: %f\n", cycles);
    }

    // Cleanup
    free(h_data);
    cudaFree(d_data);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
