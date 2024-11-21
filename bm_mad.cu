#include <stdio.h>
#include <cuda.h>

// Define the number of iterations
#define N 1e7

// Kernel for testing mad.f32
__global__ void microbenchmark(float *input1, float *input2, float *output) {
    float temp = *input1;

    for (int i = 0; i < N; i++) {
        temp = temp * (*input2) + (*input2); // mad.f32 computation
    }
    *output = temp;
}

int main(int argc, char *argv[]) {
    // Ensure correct input arguments
    if (argc != 3) {
        printf("Usage: <freq in GHz> <GPU id>\n");
        return 1;
    }

    // Parse input arguments
    float freq_ghz = atof(argv[1]); // GPU frequency in GHz
    int gpu_id = atoi(argv[2]);    // GPU device ID
    printf("Selected GPU ID: %d\n", gpu_id);
    printf("GPU frequency: %f GHz\n", freq_ghz);

    // Set the selected GPU device
    cudaSetDevice(gpu_id);

    // Host and device variables
    float *d_input1, *d_input2, *d_output;
    float h_input1 = 1.0f, h_input2 = 1.234f, h_output;

    // Allocate device memory
    cudaMalloc(&d_input1, sizeof(float));
    cudaMalloc(&d_input2, sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input values to device
    cudaMemcpy(d_input1, &h_input1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, &h_input2, sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the kernel and record time
    cudaEventRecord(start);
    microbenchmark<<<1, 1>>>(d_input1, d_input2, d_output); // Single thread
    cudaEventRecord(stop);

    // Wait for the kernel to complete and synchronize
    cudaEventSynchronize(stop);

    // Get elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory and destroy events
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print results
    printf("Time elapsed: %f ms\n", milliseconds);
    float cycles = (milliseconds / 1000.0) * freq_ghz * 1e9 / N; // Convert to cycles
    printf("mad.f32 delay: %f cycles\n", cycles);

    return 0;
}
