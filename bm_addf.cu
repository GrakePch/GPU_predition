#include <stdio.h>
#include <cuda.h>

__global__ void microbenchmarkAdd(float *input, float *output, int N) {
    float temp = *input;
    for (int i = 0; i < N; i++) {
        temp += 1.345f; // Example computation
    }
    *output = temp;
}

int main(int argc, char *argv[]) {
    
    if (argc != 2) {
        printf("usage:  gpu_frequency in Ghz\n");
        exit(1);
    }

    
    float f = atof(argv[1]);
    printf("GPU frequency %f Ghz.\n", f);

    const int N = 1e7; // Number of iterations
    float *d_input, *d_output;
    float h_input = 1.0f, h_output;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(deviceCount-1);

    cudaMalloc(&d_input, sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, &h_input, sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    microbenchmarkAdd<<<1, 1>>>(d_input, d_output, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Time elapsed: %f ms\n", milliseconds);
    float cycles = milliseconds / 1000 / N * f * 1e9;
    printf("Cycle: %f\n", cycles);
    return 0;
}
