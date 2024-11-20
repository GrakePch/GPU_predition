#include <cuda.h>
#include <stdio.h>

#define N (64 * 1024 * 1024)  // array length

__global__ void measure_memory_load_cycles(volatile float *data, float *index, unsigned long long *cycle_diff) {
    int idx = 0;

    unsigned long long start = clock64();

    idx = (int)index[idx];  // Pointer chasing

    unsigned long long end = clock64();

    *cycle_diff = end - start;

    // Prevent compiler optimization
    data[0] = idx;
}

int main() {
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(deviceCount-1);

    float *h_index;
    float *d_index, *d_data;
    unsigned long long *d_cycle_diff;
    unsigned long long cycle_diff;

    h_index = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_index[i] = (i * 37) % N;  // Randomize access pattern
    }

    cudaMalloc(&d_index, N * sizeof(float));
    cudaMalloc(&d_data, sizeof(float));
    cudaMalloc(&d_cycle_diff, sizeof(unsigned long long));

    // Load from global mem
    cudaMemcpy(d_index, h_index, N * sizeof(float), cudaMemcpyHostToDevice);
    measure_memory_load_cycles<<<1, 1>>>(d_data, d_index, d_cycle_diff);
    cudaMemcpy(&cycle_diff, d_cycle_diff, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    printf("Global memory load latency: %llu cycles\n", cycle_diff);

    cudaFree(d_index);
    cudaFree(d_data);
    cudaFree(d_cycle_diff);
    free(h_index);

    return 0;
}
