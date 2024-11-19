#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void memoryBandwidthKernel(float *dst, const float *src, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

void testBandwidth(size_t size_in_mb) {
    size_t size = size_in_mb * 1024 * 1024; // Convert MB to bytes
    size_t num_elements = size / sizeof(float);

    float *h_src = (float *)malloc(size);
    float *h_dst = (float *)malloc(size);
    if (!h_src || !h_dst) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < num_elements; ++i) {
        h_src[i] = (float)i;
    }

    float *d_src, *d_dst;
    cudaMalloc((void **)&d_src, size);
    cudaMalloc((void **)&d_dst, size);

    cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // Warm-up run
    // To avoid initialization overhead, and get some cache.
    memoryBandwidthKernel<<<blocks_per_grid, threads_per_block>>>(d_dst, d_src, num_elements);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    memoryBandwidthKernel<<<blocks_per_grid, threads_per_block>>>(d_dst, d_src, num_elements);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    float bandwidth = (2.0f * size) / (elapsed_ms * 1e6); // GB/s
    printf("Bandwidth: %.2f GB/s\n", bandwidth);

    free(h_src);
    free(h_dst);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    size_t test_size_mb = 1024; // Test with 1GB of data
    testBandwidth(test_size_mb);
    return 0;
}

