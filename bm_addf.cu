#include <stdio.h>
#include <cuda.h>

__global__ void microbenchmarkAdd(float *input, float *output, int N) {
    // float temp = *input;
    // for (int i = 0; i < N; i++) {
    //     temp /= 1.345f; // Example computation
    // }
    // *output = temp;
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

    // Allocate device memory
    cudaMalloc(&d_input, sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, &h_input, sizeof(float), cudaMemcpyHostToDevice);

    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch the kernel
    microbenchmarkAdd<<<1024000, 1024>>>(d_input, d_output, N);

    // Record the stop event
    cudaEventRecord(stop);

    // Wait for the event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the result back
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory and destroy events
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Time elapsed: %f ms\n", milliseconds);
    float cycles = milliseconds / 1000 / N * f * 1e9;
    printf("Cycle: %f\n", cycles);
    return 0;
}
