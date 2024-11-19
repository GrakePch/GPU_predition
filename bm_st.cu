#include <cuda.h>
#include <stdio.h>

#define N (64 * 1024 * 1024)  // array length

__global__ void measure_memory_store_cycles(volatile int *data, int value, unsigned long long *cycle_diff) {
    unsigned long long start = clock64();

    data[0] = value;

    unsigned long long end = clock64();
    *cycle_diff = end - start;
}

int main() {
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(deviceCount-1);

    int *h_index;
    int *d_index, *d_data;
    unsigned long long *d_cycle_diff;
    unsigned long long cycle_diff;

    h_index = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_index[i] = (i * 37) % N;  // Randomize access pattern
    }

    cudaMalloc(&d_index, N * sizeof(int));
    cudaMalloc(&d_data, sizeof(int));
    cudaMalloc(&d_cycle_diff, sizeof(unsigned long long));

    // Store to global mem
    cudaMemcpy(d_index, h_index, N * sizeof(int), cudaMemcpyHostToDevice);
    measure_memory_store_cycles<<<1, 1>>>(d_data, 2, d_cycle_diff);
    cudaMemcpy(&cycle_diff, d_cycle_diff, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    printf("Global memory store latency: %llu cycles\n", cycle_diff);

    cudaFree(d_index);
    cudaFree(d_data);
    cudaFree(d_cycle_diff);
    free(h_index);

    return 0;
}
