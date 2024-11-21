#include <stdio.h>
#include <cuda.h>

// Define the number of iterations
#define N 1e7 

// Kernel for testing bitwise AND operation
__global__ void microbenchmark(unsigned int *input, unsigned int *output) {
    unsigned int temp = *input;
    for (int i = 0; i < N; i++) {
        temp = (temp & 0xFFFF) ^ 0x1234;  // Add more operations for complexity
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
    int gpu_id = atoi(argv[2]);    // GPU ID
    printf("Selected GPU ID: %d\n", gpu_id);
    printf("GPU frequency: %f GHz\n", freq_ghz);

    // Set the selected GPU device
    cudaSetDevice(gpu_id);

    // Host and device variables
    unsigned int *d_input, *d_output;
    unsigned int h_input = 0xFFFFFFFF, h_output;

    // Allocate device memory
    cudaMalloc(&d_input, sizeof(unsigned int));
    cudaMalloc(&d_output, sizeof(unsigned int));

    // Copy input value to device
    cudaMemcpy(d_input, &h_input, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the kernel and record time
    cudaEventRecord(start);
    microbenchmark<<<1, 1>>>(d_input, d_output); // Single-thread kernel
    cudaEventRecord(stop);

    // Wait for the kernel to complete and synchronize
    cudaEventSynchronize(stop);

    // Get elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(&h_output, d_output, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Free memory and destroy events
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print results
    printf("Time elapsed: %f ms\n", milliseconds);
    float cycles = (milliseconds / 1000.0) * freq_ghz * 1e9 / N; // Convert to cycles
    printf("and delay: %f cycles\n", cycles);

    return 0;
}
