#include <cuda.h>
#include <stdio.h>

#define N (64 * 1024 * 1024)  // array length

__global__ void measure_memory_load_cycles(volatile int *data, int *index, unsigned long long *cycle_diff) {
    int idx = 0;

    unsigned long long start = clock64();

    idx = index[idx];  // Pointer chasing

    unsigned long long end = clock64();

    *cycle_diff = end - start;

    // Prevent compiler optimization
    data[0] = idx;
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

    // Load from global mem
    cudaMemcpy(d_index, h_index, N * sizeof(int), cudaMemcpyHostToDevice);
    measure_memory_load_cycles<<<1, 1>>>(d_data, d_index, d_cycle_diff);
    cudaMemcpy(&cycle_diff, d_cycle_diff, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    printf("Global memory load latency: %llu cycles\n", cycle_diff);

    cudaFree(d_index);
    cudaFree(d_data);
    cudaFree(d_cycle_diff);
    free(h_index);

    return 0;
}
